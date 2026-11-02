//! Speaker identification: WeSpeaker ResNet embeddings of short speech windows, clustered into speakers

use crate::audio::PcmReader;
use crate::ggml::{Backend, F16, F32, Graph, T, Weights};
use candle_core::pickle::PthTensors;
use std::path::Path;

/// Embedding window and hop (seconds), at 16 kHz
const WIN: f64 = 1.5;
const HOP: f64 = 0.75;
const RATE: f64 = 16_000.0;
const MELS: usize = 80;
/// Windows per graph run
const BATCH: usize = 32;

struct Conv {
    w: T,
    b: T,
    stride: i32,
    pad: i32,
}

struct Block {
    convs: Vec<Conv>,
    shortcut: Option<Conv>,
}

pub struct Embedder {
    // fields drop in order: the weights buffer must go before its backend
    _weights: Weights,
    backend: Backend,
    stem: Conv,
    blocks: Vec<Block>,
    /// final linear layer, run on the CPU: [out][in]
    seg_w: Vec<f32>,
    seg_b: Vec<f32>,
    dim: usize,
}

impl Embedder {
    /// Loads a WeSpeaker ResNet checkpoint (ResNet34 / 221 / 293 …), folding batch norms into convs
    pub fn load(path: &Path, gpu: bool) -> Result<Self, String> {
        let pth = PthTensors::new(path, None).map_err(|e| format!("cannot read speaker model: {e}"))?;
        let get = |n: &str| -> Result<Vec<f32>, String> {
            let t = pth.get(n).map_err(|e| e.to_string())?.ok_or_else(|| format!("speaker model: missing {n}"))?;
            t.flatten_all().and_then(|t| t.to_dtype(candle_core::DType::F32)?.to_vec1()).map_err(|e| e.to_string())
        };
        let shape = |n: &str| -> Result<Vec<i64>, String> {
            let i = pth.tensor_infos().get(n).ok_or_else(|| format!("speaker model: missing {n}"))?;
            Ok(i.layout.shape().dims().iter().map(|&d| d as i64).collect())
        };
        let has = |n: &str| pth.tensor_infos().contains_key(n);

        // (ggml tensor, folded weight, folded bias) for every conv + batch norm pair
        let mut weights = Weights::new(2048);
        let mut uploads: Vec<(T, Vec<f32>)> = Vec::new();
        let mut conv = |w: &mut Weights, conv: &str, bn: &str, stride: i32| -> Result<Conv, String> {
            let s = shape(conv)?; // (out, in, kh, kw)
            let mut wt = get(conv)?;
            let (gamma, beta, mean, var) = (get(&format!("{bn}.weight"))?, get(&format!("{bn}.bias"))?, get(&format!("{bn}.running_mean"))?, get(&format!("{bn}.running_var"))?);
            let per = (s[1] * s[2] * s[3]) as usize;
            let mut bias = vec![0f32; s[0] as usize];
            for o in 0..s[0] as usize {
                let k = gamma[o] / (var[o] + 1e-5).sqrt();
                wt[o * per..(o + 1) * per].iter_mut().for_each(|x| *x *= k);
                bias[o] = beta[o] - mean[o] * k;
            }
            let wten = w.tensor(F16, &[s[3], s[2], s[1], s[0]]);
            let bten = w.tensor(F32, &[1, 1, s[0]]);
            uploads.push((wten, wt));
            uploads.push((bten, bias));
            Ok(Conv { w: wten, b: bten, stride, pad: (s[3] / 2) as i32 })
        };

        let stem = conv(&mut weights, "conv1.weight", "bn1", 1)?;
        let bottleneck = has("layer1.0.conv3.weight");
        let mut blocks = Vec::new();
        for layer in 1..=4 {
            for i in 0.. {
                let p = format!("layer{layer}.{i}");
                if !has(&format!("{p}.conv1.weight")) {
                    break;
                }
                let stride = if i == 0 && layer > 1 { 2 } else { 1 };
                let names: &[&str] = if bottleneck { &["1", "2", "3"] } else { &["1", "2"] };
                let mut convs = Vec::new();
                for (j, n) in names.iter().enumerate() {
                    let s = if (bottleneck && j == 1) || (!bottleneck && j == 0) { stride } else { 1 };
                    convs.push(conv(&mut weights, &format!("{p}.conv{n}.weight"), &format!("{p}.bn{n}"), s)?);
                }
                let shortcut = if has(&format!("{p}.shortcut.0.weight")) {
                    Some(conv(&mut weights, &format!("{p}.shortcut.0.weight"), &format!("{p}.shortcut.1"), stride)?)
                } else {
                    None
                };
                blocks.push(Block { convs, shortcut });
            }
        }
        let seg_shape = shape("seg_1.weight")?;
        let (seg_w, seg_b) = (get("seg_1.weight")?, get("seg_1.bias")?);

        let backend = Backend::new(gpu);
        weights.alloc(&backend)?;
        for (t, v) in &uploads {
            weights.set(*t, v);
        }
        Ok(Self { _weights: weights, backend, stem, blocks, seg_w, seg_b, dim: seg_shape[0] as usize })
    }

    fn graph(&self, g: &Graph, x: T) -> T {
        let conv = |x, c: &Conv| g.add(g.conv_2d(c.w, x, [c.stride, c.stride], [c.pad, c.pad]), c.b);
        let mut x = g.relu(conv(x, &self.stem));
        for b in &self.blocks {
            let mut h = x;
            for (i, c) in b.convs.iter().enumerate() {
                h = conv(h, c);
                if i + 1 < b.convs.len() {
                    h = g.relu(h);
                }
            }
            let short = b.shortcut.as_ref().map_or(x, |c| conv(x, c));
            x = g.relu(g.add(h, short));
        }
        x
    }

    /// Embeddings of equally long 16 kHz windows
    pub fn embed(&self, windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::with_capacity(windows.len());
        for batch in windows.chunks(BATCH) {
            let feats: Vec<Vec<[f32; MELS]>> = batch.iter().map(|w| cmn(fbank(w))).collect();
            let frames = feats[0].len();
            let n = batch.len();
            // ggml input [time, mel, 1, batch]
            let mut data = vec![0f32; frames * MELS * n];
            for (b, f) in feats.iter().enumerate() {
                for (t, row) in f.iter().enumerate().take(frames) {
                    for (m, &v) in row.iter().enumerate() {
                        data[(b * MELS + m) * frames + t] = v;
                    }
                }
            }
            let mut shape = [0i64; 4];
            let maps: Vec<f32> = self.backend.run(|g| {
                let input = g.input(F32, &[frames as i64, MELS as i64, 1, n as i64]);
                let y = self.graph(g, input);
                shape = [y.ne(0), y.ne(1), y.ne(2), y.ne(3)];
                (y, vec![(input, crate::ggml::f32_bytes(&data))])
            })?;
            // temporal statistics pooling (mean and unbiased std over time) + linear, per window
            let [t, f, c, _] = shape.map(|v| v as usize);
            for b in 0..n {
                let map = &maps[b * t * f * c..(b + 1) * t * f * c];
                let mut stats = vec![0f32; 2 * f * c];
                for (row, s) in map.chunks(t).enumerate() {
                    let mean = s.iter().sum::<f32>() / t as f32;
                    let var = s.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / (t.max(2) - 1) as f32;
                    stats[row] = mean;
                    stats[f * c + row] = (var + 1e-7).sqrt();
                }
                let emb: Vec<f32> = (0..self.dim)
                    .map(|o| self.seg_b[o] + self.seg_w[o * stats.len()..(o + 1) * stats.len()].iter().zip(&stats).map(|(a, b)| a * b).sum::<f32>())
                    .collect();
                out.push(emb);
            }
        }
        Ok(out)
    }
}

fn fbank(wav: &[f32]) -> Vec<[f32; MELS]> {
    const LEN: usize = 400;
    const SHIFT: usize = 160;
    const NFFT: usize = 512;
    if wav.len() < LEN {
        return Vec::new();
    }
    let window: Vec<f32> = (0..LEN).map(|i| (0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (LEN - 1) as f32).cos()).powf(0.85)).collect();
    let banks = mel_banks(NFFT);
    let frames = (wav.len() - LEN) / SHIFT + 1;
    let mut out = Vec::with_capacity(frames);
    let (mut re, mut im) = (vec![0f32; NFFT], vec![0f32; NFFT]);
    for f in 0..frames {
        let x: Vec<f32> = wav[f * SHIFT..f * SHIFT + LEN].iter().map(|s| s * 32768.0).collect();
        let mean = x.iter().sum::<f32>() / LEN as f32;
        re.iter_mut().for_each(|v| *v = 0.0);
        im.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..LEN {
            let prev = if i == 0 { x[0] } else { x[i - 1] } - mean;
            re[i] = ((x[i] - mean) - 0.97 * prev) * window[i];
        }
        fft(&mut re, &mut im);
        let power: Vec<f32> = (0..=NFFT / 2).map(|k| re[k] * re[k] + im[k] * im[k]).collect();
        let mut row = [0f32; MELS];
        for (m, (start, weights)) in banks.iter().enumerate() {
            let e: f32 = weights.iter().enumerate().map(|(j, w)| w * power[start + j]).sum();
            row[m] = e.max(f32::EPSILON).ln();
        }
        out.push(row);
    }
    out
}

fn cmn(mut feats: Vec<[f32; MELS]>) -> Vec<[f32; MELS]> {
    let n = feats.len().max(1) as f32;
    for m in 0..MELS {
        let mean = feats.iter().map(|r| r[m]).sum::<f32>() / n;
        feats.iter_mut().for_each(|r| r[m] -= mean);
    }
    feats
}

fn mel_banks(nfft: usize) -> Vec<(usize, Vec<f32>)> {
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let (lo, hi) = (mel(20.0), mel(RATE / 2.0));
    let delta = (hi - lo) / (MELS + 1) as f64;
    let bin_hz = RATE / nfft as f64;
    (0..MELS)
        .map(|m| {
            let (l, c, r) = (lo + m as f64 * delta, lo + (m + 1) as f64 * delta, lo + (m + 2) as f64 * delta);
            let mut first = usize::MAX;
            let mut w = Vec::new();
            for k in 0..nfft / 2 {
                let x = mel(k as f64 * bin_hz);
                let v = if x > l && x < r { if x <= c { (x - l) / (c - l) } else { (r - x) / (r - c) } } else { 0.0 };
                if v > 0.0 {
                    first = first.min(k);
                    w.resize(k - first + 1, 0.0);
                    w[k - first] = v as f32;
                }
            }
            (first.min(nfft / 2), w)
        })
        .collect()
}

fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (ang * k as f32).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (tr, ti) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

pub fn diarize(model: &Embedder, reader: &mut PcmReader, vad: &[f32], speakers: Option<usize>, progress: &mut dyn FnMut(f32) -> bool) -> Result<Vec<(f64, f64, u32)>, String> {
    let win = (WIN * RATE) as usize;
    let total = reader.asr_len();
    let speech = |a: f64, b: f64| {
        let (a, b) = ((a * 100.0) as usize, ((b * 100.0) as usize).min(vad.len()));
        vad.get(a..b).map_or(0.0, |v| v.iter().filter(|&&p| p > 0.35).count() as f64 / (b - a).max(1) as f64)
    };
    let mut windows: Vec<Window> = Vec::new();
    for (a, b) in crate::transcribe::speech_regions(vad, 10, 30) {
        let (a, b) = (a as f64 / 100.0, b as f64 / 100.0);
        if b - a < 0.4 {
            continue;
        }
        let mut t = a;
        loop {
            let end = (t + WIN).min(b);
            let start = (end - WIN).max(a);
            if speech(start, end) >= 0.6 {
                windows.push(Window { start, end, first: start <= a, last: end >= b });
            }
            if end >= b {
                break;
            }
            t += HOP;
        }
    }
    if windows.is_empty() {
        return Ok(Vec::new());
    }
    let mut embs: Vec<Vec<f32>> = Vec::with_capacity(windows.len());
    for (i, batch) in windows.chunks(BATCH * 4).enumerate() {
        let audio: Vec<Vec<f32>> = batch
            .iter()
            .map(|w| {
                let a = ((w.start * RATE) as usize).min(total);
                let len = (((w.end - w.start) * RATE) as usize).min(total - a).max(1);
                let x = reader.read_asr(a, len);
                x.iter().cycle().take(win).copied().collect()
            })
            .collect();
        embs.extend(model.embed(&audio)?);
        if !progress(((i + 1) * BATCH * 4).min(windows.len()) as f32 / windows.len() as f32) {
            return Err(crate::i18n::tr("cancelled").into());
        }
    }
    let labels = cluster(&embs, speakers);

    let mut turns: Vec<(f64, f64, u32)> = Vec::new();
    for (w, &l) in windows.iter().zip(&labels) {
        let c = (w.start + w.end) / 2.0;
        let s = if w.first { w.start } else { c - HOP / 2.0 };
        let e = if w.last { w.end } else { c + HOP / 2.0 };
        match turns.last_mut() {
            Some(last) if last.2 == l as u32 && s - last.1 < 1.0 => last.1 = e,
            _ => turns.push((s, e, l as u32)),
        }
    }
    let mut i = 0;
    while turns.len() > 1 && i < turns.len() {
        if turns[i].1 - turns[i].0 < 1.0 {
            let t = turns.remove(i);
            if i > 0 {
                turns[i - 1].1 = turns[i - 1].1.max(t.1);
            } else {
                turns[0].0 = t.0;
            }
            if i > 0 && i < turns.len() && turns[i - 1].2 == turns[i].2 {
                let n = turns.remove(i);
                turns[i - 1].1 = n.1;
            }
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
    Ok(turns)
}

struct Window {
    start: f64,
    end: f64,
    first: bool,
    last: bool,
}

pub fn speaker_at(turns: &[(f64, f64, u32)], t: f64) -> Option<u32> {
    let i = turns.partition_point(|x| x.0 <= t);
    let before = i.checked_sub(1).map(|j| turns[j]);
    match (before, turns.get(i)) {
        (Some(b), _) if t <= b.1 => Some(b.2),
        (Some(b), Some(a)) => Some(if t - b.1 <= a.0 - t { b.2 } else { a.2 }),
        (Some(b), None) => Some(b.2),
        (None, Some(a)) => Some(a.2),
        (None, None) => None,
    }
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Cosine similarity below which clusters stay apart when the speaker count is unknown
const THRESHOLD: f32 = 0.35;

/// Agglomerative clustering on a subsample, then k-means refinement over all embeddings
fn cluster(embs: &[Vec<f32>], speakers: Option<usize>) -> Vec<usize> {
    let e: Vec<Vec<f32>> = embs.iter().map(|v| normalize(v)).collect();
    if e.len() < 2 {
        return vec![0; e.len()];
    }
    let step = e.len().div_ceil(1000);
    let sample: Vec<usize> = (0..e.len()).step_by(step).collect();
    let mut sums: Vec<Vec<f32>> = sample.iter().map(|&i| e[i].clone()).collect();
    let mut counts = vec![1usize; sums.len()];
    let sim = |a: &[f32], b: &[f32]| dot(a, b) / (dot(a, a) * dot(b, b)).sqrt().max(1e-9);
    let mut alive: Vec<usize> = (0..sums.len()).collect();
    loop {
        let target = speakers.unwrap_or(1).max(1);
        if alive.len() <= target {
            break;
        }
        let mut best = (f32::MIN, 0, 0);
        for (x, &i) in alive.iter().enumerate() {
            for &j in &alive[x + 1..] {
                let s = sim(&sums[i], &sums[j]);
                if s > best.0 {
                    best = (s, i, j);
                }
            }
        }
        if speakers.is_none() && best.0 < THRESHOLD {
            break;
        }
        let (_, i, j) = best;
        let add = std::mem::take(&mut sums[j]);
        sums[i].iter_mut().zip(&add).for_each(|(a, b)| *a += b);
        counts[i] += counts[j];
        alive.retain(|&k| k != j);
    }
    let min = (sample.len() / 50).max(3);
    let mut centroids: Vec<Vec<f32>> = alive.iter().filter(|&&i| counts[i] >= min || speakers.is_some()).map(|&i| normalize(&sums[i])).collect();
    if centroids.is_empty() {
        centroids.push(normalize(&sums[alive[0]]));
    }
    let mut labels = vec![0usize; e.len()];
    for _ in 0..5 {
        for (l, v) in labels.iter_mut().zip(&e) {
            *l = (0..centroids.len()).max_by(|&a, &b| dot(v, &centroids[a]).total_cmp(&dot(v, &centroids[b]))).unwrap();
        }
        for (c, centroid) in centroids.iter_mut().enumerate() {
            let mut s = vec![0f32; e[0].len()];
            for (v, _) in e.iter().zip(&labels).filter(|(_, l)| **l == c) {
                s.iter_mut().zip(v).for_each(|(a, b)| *a += b);
            }
            if s.iter().any(|x| *x != 0.0) {
                *centroid = normalize(&s);
            }
        }
    }
    let mut order: Vec<usize> = Vec::new();
    for &l in &labels {
        if !order.contains(&l) {
            order.push(l);
        }
    }
    labels.iter().map(|l| order.iter().position(|o| o == l).unwrap()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft() {
        let n = 16;
        let x: Vec<f32> = (0..n).map(|i| ((i * 37) % 11) as f32 - 5.0).collect();
        let (mut re, mut im) = (x.clone(), vec![0f32; n]);
        fft(&mut re, &mut im);
        for k in 0..n {
            let (r, i) = x.iter().enumerate().fold((0f32, 0f32), |(r, i), (t, v)| {
                let a = -2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
                (r + v * a.cos(), i + v * a.sin())
            });
            assert!((re[k] - r).abs() < 1e-3 && (im[k] - i).abs() < 1e-3, "bin {k}");
        }
    }

    #[test]
    fn fbank_shape_and_tone() {
        // 1 kHz tone: energy peaks in the mel bin around 1 kHz
        let wav: Vec<f32> = (0..16000).map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16000.0).sin() * 0.3).collect();
        let f = fbank(&wav);
        assert_eq!(f.len(), (16000 - 400) / 160 + 1);
        let banks = mel_banks(512);
        let peak = (0..MELS).max_by(|&a, &b| f[50][a].total_cmp(&f[50][b])).unwrap();
        let center_hz = |m: usize| (banks[m].0 as f64 + banks[m].1.len() as f64 / 2.0) * 16000.0 / 512.0;
        assert!((center_hz(peak) - 1000.0).abs() < 120.0, "peak at {} Hz", center_hz(peak));
    }

    #[test]
    fn clustering_separates_and_orders() {
        // two speakers in alternating runs, with noise around their directions
        let a: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.0 }).collect();
        let b: Vec<f32> = (0..16).map(|i| if i < 8 { 0.0 } else { 1.0 }).collect();
        let noisy = |v: &Vec<f32>, k: usize| v.iter().enumerate().map(|(i, x)| x + ((i * 31 + k * 17) % 7) as f32 * 0.04).collect::<Vec<_>>();
        let embs: Vec<Vec<f32>> = (0..60).map(|k| if (k / 10) % 2 == 0 { noisy(&b, k) } else { noisy(&a, k) }).collect();
        let labels = cluster(&embs, None);
        for (k, l) in labels.iter().enumerate() {
            assert_eq!(*l, (k / 10) % 2, "window {k}");
        }
        assert!(cluster(&embs[..10], None).iter().all(|&l| l == 0));
        assert_eq!(cluster(&embs, Some(3)).iter().max(), Some(&2));
    }

    #[test]
    fn speaker_lookup() {
        let turns = [(0.0, 2.0, 0), (2.5, 5.0, 1)];
        assert_eq!(speaker_at(&turns, 1.0), Some(0));
        assert_eq!(speaker_at(&turns, 2.2), Some(0));
        assert_eq!(speaker_at(&turns, 2.4), Some(1));
        assert_eq!(speaker_at(&turns, 9.0), Some(1));
        assert_eq!(speaker_at(&[], 1.0), None);
    }

    /// 'LATTR_AUDIO=conv.wav LATTR_SPK_MODEL=resnet34-LM.pt cargo test --release diarize_e2e -- --ignored --nocapture'
    /// Optional: 'LATTR_SPEAKERS=n', 'LATTR_GPU=1', 'LATTR_DIAR_OUT=turns.txt'
    #[test]
    #[ignore]
    fn diarize_e2e() {
        use crate::audio::{LoadMsg, LoadProgress, load};
        let (tx, rx) = std::sync::mpsc::channel();
        load(std::env::var("LATTR_AUDIO").unwrap().into(), tx, std::sync::Arc::new(LoadProgress::default()), || {});
        let (mut orig, mut vad) = (None, None);
        while orig.is_none() || vad.is_none() {
            match rx.recv().unwrap() {
                LoadMsg::Original(o) => orig = Some(o),
                LoadMsg::Denoised(_, v) => vad = Some(v),
                LoadMsg::Error(e) => panic!("{e}"),
            }
        }
        let vad = vad.unwrap();
        let t = std::time::Instant::now();
        let model = Embedder::load(std::env::var("LATTR_SPK_MODEL").unwrap().as_ref(), std::env::var_os("LATTR_GPU").is_some()).unwrap();
        println!("loaded in {:?} (gpu {})", t.elapsed(), model.backend.gpu);
        let speakers = std::env::var("LATTR_SPEAKERS").ok().and_then(|v| v.parse().ok());
        let t = std::time::Instant::now();
        let turns = diarize(&model, &mut orig.unwrap().reader().unwrap(), &vad, speakers, &mut |_| true).unwrap();
        println!("diarized in {:?}: {} turns", t.elapsed(), turns.len());
        let text: String = turns.iter().map(|(a, b, s)| format!("{a:.2} {b:.2} {s}\n")).collect();
        if let Ok(out) = std::env::var("LATTR_DIAR_OUT") {
            std::fs::write(out, &text).unwrap();
        }
    }
}
