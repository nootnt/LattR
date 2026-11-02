//! Meta's Omnilingual ASR CTC models (wav2vec2 encoder + CTC head) on ggml (Vulkan or CPU). Loads the official fairseq2 checkpoints (read with candle's PyTorch loader)

use crate::ggml::{self, Backend, F16, F32, Graph, T, Weights};
use crate::transcribe::Word;
use candle_core::pickle::PthTensors;
use std::path::Path;

/// Feature extractor downsampling: 16 kHz samples per output frame (20 ms)
const FRAME: usize = 320;
/// Segment length - accuracy drops sharply with longer segments
pub const MAX_SECONDS: f64 = 10.0;
/// fairseq2 keeps 16 heads in every released size
const HEADS: i64 = 16;
const POS_GROUPS: i64 = 16;

struct Norm {
    w: T,
    b: T,
}

struct Lin {
    w: T,
    b: T,
}

struct Conv {
    /// [k * in, out], laid out for time-major windows, see 'conv'
    w: T,
    b: T,
    norm: Norm,
    k: i64,
    stride: i64,
}

struct Layer {
    ln1: Norm,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln2: Norm,
    fc1: Lin,
    fc2: Lin,
}

pub struct Omni {
    // fields drop in order: the weights buffer must go before its backend
    _weights: Weights,
    backend: Backend,
    convs: Vec<Conv>,
    post_ln: Norm,
    proj: Lin,
    /// weight-normed positional conv [k, in / groups, out]
    pos_w: T,
    pos_b: T,
    layers: Vec<Layer>,
    ln: Norm,
    head: Lin,
    pieces: Vec<String>,
    special: Vec<bool>,
}

/// How a checkpoint tensor is rearranged before upload.
enum Fix {
    None,
    /// conv (out, in, k) -> (out, k, in), uploaded as [k * in, out]
    ConvKIn,
    /// torch 'weight_norm(dim=2)', weight_v scaled by weight_g per kernel position
    WeightNorm,
}

/// Declare weights from the checkpoint's shape table, then upload them one at a time
struct Loader {
    pth: PthTensors,
    weights: Weights,
    uploads: Vec<(T, String, Fix)>,
}

impl Loader {
    fn tensor(&mut self, name: &str, ty: ggml::Type, fix: Fix) -> Result<T, String> {
        let info = self.pth.tensor_infos().get(name).ok_or_else(|| format!("unexpected checkpoint layout: missing {name}"))?;
        // torch lists dimensions slowest first, ggml fastest first
        let mut ne: Vec<i64> = info.layout.shape().dims().iter().rev().map(|&d| d as i64).collect();
        if let Fix::ConvKIn = fix {
            ne = vec![ne[0] * ne[1], ne[2]];
        }
        let t = self.weights.tensor(ty, &ne);
        self.uploads.push((t, name.to_string(), fix));
        Ok(t)
    }

    fn norm(&mut self, p: &str) -> Result<Norm, String> {
        Ok(Norm { w: self.tensor(&format!("{p}.weight"), F32, Fix::None)?, b: self.tensor(&format!("{p}.bias"), F32, Fix::None)? })
    }

    fn lin(&mut self, p: &str) -> Result<Lin, String> {
        Ok(Lin { w: self.tensor(&format!("{p}.weight"), F16, Fix::None)?, b: self.tensor(&format!("{p}.bias"), F32, Fix::None)? })
    }

    fn upload(mut self, backend: &Backend) -> Result<Weights, String> {
        self.weights.alloc(backend)?;
        let get = |n: &str| -> Result<candle_core::Tensor, String> {
            self.pth.get(n).map_err(|e| e.to_string())?.ok_or_else(|| format!("missing {n}"))
        };
        for (t, name, fix) in &self.uploads {
            let x = get(name)?;
            let x = match fix {
                Fix::None => x,
                Fix::ConvKIn => x.permute((0, 2, 1)).and_then(|x| x.contiguous()).map_err(|e| e.to_string())?,
                Fix::WeightNorm => {
                    let g = get(&name.replace("weight_v", "weight_g"))?;
                    (|| x.broadcast_mul(&g.broadcast_div(&x.sqr()?.sum_keepdim(0)?.sum_keepdim(1)?.sqrt()?)?))().map_err(|e| e.to_string())?
                }
            };
            let v: Vec<f32> = x.flatten_all().and_then(|x| x.to_dtype(candle_core::DType::F32)?.to_vec1()).map_err(|e| e.to_string())?;
            self.weights.set(*t, &v);
        }
        Ok(self.weights)
    }
}

impl Omni {
    /// Loads an 'omniASR-CTC-*.pt' checkpoint with its tokenizer '.model'
    pub fn load(checkpoint: &Path, gpu: bool) -> Result<Self, String> {
        let tokenizer = std::fs::read_dir(checkpoint.parent().ok_or("bad model path")?)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "model"))
            .ok_or("tokenizer (.model) not found next to the checkpoint")?;
        let (pieces, special) = read_sentencepiece(&std::fs::read(tokenizer).map_err(|e| e.to_string())?)?;
        let pth = PthTensors::new(checkpoint, Some("model")).map_err(|e| format!("cannot read checkpoint: {e}"))?;
        let mut l = Loader { pth, weights: Weights::new(1024), uploads: Vec::new() };

        let fe = "encoder_frontend.feature_extractor.layers";
        let mut convs = Vec::new();
        for (i, stride) in [5, 2, 2, 2, 2, 2, 2].into_iter().enumerate() {
            let k = *l.pth.tensor_infos().get(&format!("{fe}.{i}.conv.weight")).ok_or("missing conv")?.layout.shape().dims().last().unwrap() as i64;
            convs.push(Conv {
                w: l.tensor(&format!("{fe}.{i}.conv.weight"), F16, Fix::ConvKIn)?,
                b: l.tensor(&format!("{fe}.{i}.conv.bias"), F32, Fix::None)?,
                norm: l.norm(&format!("{fe}.{i}.layer_norm"))?,
                k,
                stride,
            });
        }
        let post_ln = l.norm("encoder_frontend.post_extract_layer_norm")?;
        let proj = l.lin("encoder_frontend.model_dim_proj")?;
        let pos_w = l.tensor("encoder_frontend.pos_encoder.conv.weight_v", F16, Fix::WeightNorm)?;
        let pos_b = l.tensor("encoder_frontend.pos_encoder.conv.bias", F32, Fix::None)?;
        let mut layers = Vec::new();
        for i in 0.. {
            let p = format!("encoder.layers.{i}");
            if !l.pth.tensor_infos().contains_key(&format!("{p}.self_attn.q_proj.weight")) {
                break;
            }
            layers.push(Layer {
                ln1: l.norm(&format!("{p}.self_attn_layer_norm"))?,
                q: l.lin(&format!("{p}.self_attn.q_proj"))?,
                k: l.lin(&format!("{p}.self_attn.k_proj"))?,
                v: l.lin(&format!("{p}.self_attn.v_proj"))?,
                o: l.lin(&format!("{p}.self_attn.output_proj"))?,
                ln2: l.norm(&format!("{p}.ffn_layer_norm"))?,
                fc1: l.lin(&format!("{p}.ffn.inner_proj"))?,
                fc2: l.lin(&format!("{p}.ffn.output_proj"))?,
            });
        }
        let ln = l.norm("encoder.layer_norm")?;
        let head = l.lin("final_proj")?;
        let backend = Backend::new(gpu);
        let weights = l.upload(&backend)?;
        Ok(Self { _weights: weights, backend, convs, post_ln, proj, pos_w, pos_b, layers, ln, head, pieces, special })
    }

    pub fn on_gpu(&self) -> bool {
        self.backend.gpu
    }

    /// Words with times relative to the start of 'wav' (16 kHz)
    pub fn transcribe(&self, wav: &[f32]) -> Result<Vec<Word>, String> {
        if wav.len() < FRAME * 2 {
            return Ok(Vec::new());
        }
        let ids: Vec<i32> = self.backend.run(|g| {
            let (input, data) = normalized_input(g, wav);
            (g.argmax(self.logits(g, input)), vec![(input, data)])
        })?;
        let ids: Vec<u32> = ids.into_iter().map(|i| i as u32).collect();
        Ok(ctc_words(&ids, &self.pieces, &self.special))
    }

    /// Per-frame CTC log-probabilities of 'wav' (16 kHz), processed in 'MAX_SECONDS' pieces
    /// Returns (frames × vocabulary) row-major
    pub fn emissions(&self, wav: &[f32]) -> Result<Vec<f32>, String> {
        let v = self.pieces.len();
        let piece = (MAX_SECONDS * 16_000.0) as usize;
        let mut out = Vec::new();
        for w in wav.chunks(piece) {
            let frames = (w.len().saturating_sub(400)) / FRAME + 1;
            if w.len() < FRAME * 2 {
                out.extend(std::iter::repeat_n(-(v as f32).ln(), frames * v));
                continue;
            }
            let logits: Vec<f32> = self.backend.run(|g| {
                let (input, data) = normalized_input(g, w);
                (self.logits(g, input), vec![(input, data)])
            })?;
            for row in logits.chunks(v) {
                let max = row.iter().fold(f32::MIN, |m, &x| m.max(x));
                let lse = max + row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
                out.extend(row.iter().map(|x| x - lse));
            }
        }
        Ok(out)
    }

    pub fn pieces(&self) -> (&[String], &[bool]) {
        (&self.pieces, &self.special)
    }

    /// Encoder + CTC head: logits [vocabulary, frames]. Activations are [channels, frames]
    fn logits(&self, g: &Graph, wav: T) -> T {
        let norm = |x, n: &Norm| g.layer_norm(x, n.w, n.b, 1e-5);
        let mut x = wav;
        for c in &self.convs {
            x = g.add(conv(g, x, c.w, c.k, c.stride), c.b);
            x = g.gelu(norm(x, &c.norm));
        }
        let x = g.linear(norm(x, &self.post_ln), self.proj.w, Some(self.proj.b));
        let (dim, len) = (x.ne(0), x.ne(1));

        // grouped positional conv: per group, im2col over time then one matmul
        let xt = g.cont(g.transpose(x)); // [frames, channels]
        let (k, per) = (self.pos_w.ne(0), self.pos_w.ne(1));
        let mut pos: Option<T> = None;
        for grp in 0..POS_GROUPS {
            let xg = g.view_2d(xt, len, per, xt.nb(1), (grp * per) as usize * xt.nb(1));
            let kg = g.view_3d(self.pos_w, [k, per, per], self.pos_w.nb(1), self.pos_w.nb(2), (grp * per) as usize * self.pos_w.nb(2));
            let cols = g.im2col_1d(kg, xg, 1, (k / 2) as i32); // [per * k, len + 1]
            let y = g.mul_mat(g.reshape_2d(kg, k * per, per), cols); // [per, len + 1]
            pos = Some(pos.map_or(y, |p| g.concat(p, y, 0)));
        }
        let pos = pos.unwrap();
        // even kernel: drop the extra frame
        let pos = g.cont(g.view_2d(pos, dim, len, pos.nb(1), 0));
        let mut x = g.add(x, g.gelu(g.add(pos, self.pos_b)));

        let hd = dim / HEADS;
        let scale = 1.0 / (hd as f32).sqrt();
        for l in &self.layers {
            let h = norm(x, &l.ln1);
            // [hd, heads, frames] -> [hd, frames, heads]
            let split = |t| g.cont(g.permute(g.reshape_3d(t, hd, HEADS, len), [0, 2, 1, 3]));
            let q = split(g.linear(h, l.q.w, Some(l.q.b)));
            let kk = split(g.linear(h, l.k.w, Some(l.k.b)));
            // values as [frames, hd, heads]
            let v = g.cont(g.permute(g.reshape_3d(g.linear(h, l.v.w, Some(l.v.b)), hd, HEADS, len), [1, 2, 0, 3]));
            let att = g.soft_max(g.mul_mat(kk, q), scale); // [frames_k, frames_q, heads]
            let a = g.mul_mat(v, att); // [hd, frames_q, heads]
            let a = g.reshape_2d(g.cont(g.permute(a, [0, 2, 1, 3])), dim, len);
            x = g.add(x, g.linear(a, l.o.w, Some(l.o.b)));
            let h = norm(x, &l.ln2);
            let h = g.gelu(g.linear(h, l.fc1.w, Some(l.fc1.b)));
            x = g.add(x, g.linear(h, l.fc2.w, Some(l.fc2.b)));
        }
        g.linear(norm(x, &self.ln), self.head.w, Some(self.head.b))
    }
}

/// Input tensor for 'wav', normalized to zero mean and unit variance
fn normalized_input(g: &Graph, wav: &[f32]) -> (T, Vec<u8>) {
    let n = wav.len() as f32;
    let mean = wav.iter().sum::<f32>() / n;
    let var = wav.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    let norm: Vec<f32> = wav.iter().map(|x| (x - mean) * inv).collect();
    (g.input(F32, &[1, wav.len() as i64]), ggml::f32_bytes(&norm))
}

/// Strided 1D convolution as one matmul, requires 'stride <= k <= 2 * stride'
fn conv(g: &Graph, x: T, w: T, k: i64, stride: i64) -> T {
    let (c, len) = (x.ne(0), x.ne(1));
    assert!(k >= stride && k <= 2 * stride, "unsupported kernel {k} / stride {stride}");
    let out = (len - k) / stride + 1;
    let rows = if k > stride { out + 1 } else { out };
    let x = if rows * stride > len {
        g.pad(x, [0, (rows * stride - len) as i32, 0, 0])
    } else {
        g.view_2d(x, c, rows * stride, x.nb(1), 0)
    };
    let a = g.reshape_2d(g.cont(x), stride * c, rows);
    let cols = if k > stride {
        let first = g.view_2d(a, stride * c, out, a.nb(1), 0);
        let next = g.view_2d(a, (k - stride) * c, out, a.nb(1), a.nb(1));
        g.concat(g.cont(first), g.cont(next), 0)
    } else {
        a
    };
    g.mul_mat(w, cols)
}

/// Greedy CTC: collapse repeats, drop special tokens (blank included), split words at
/// spaces ("▁" or literal space, depending on the tokenizer)
fn ctc_words(ids: &[u32], pieces: &[String], special: &[bool]) -> Vec<Word> {
    let frame_secs = FRAME as f64 / 16_000.0;
    let mut words: Vec<Word> = Vec::new();
    let mut prev = u32::MAX;
    let mut open = false;
    for (f, &id) in ids.iter().enumerate() {
        if id == prev {
            continue;
        }
        prev = id;
        if special.get(id as usize).copied().unwrap_or(true) {
            continue;
        }
        let t = f as f64 * frame_secs;
        for c in pieces[id as usize].chars() {
            if c == '▁' || c.is_whitespace() {
                open = false;
            } else if open {
                let w = words.last_mut().unwrap();
                w.text.push(c);
                w.t1 = t + frame_secs;
            } else {
                words.push(Word { text: format!(" {c}"), t0: t, t1: t + frame_secs, speaker: None });
                open = true;
            }
        }
    }
    words
}

/// Minimal SentencePiece model reader: returns the pieces and which ones are control,
/// unknown or unused (never emitted as text).
fn read_sentencepiece(buf: &[u8]) -> std::result::Result<(Vec<String>, Vec<bool>), String> {
    fn varint(b: &[u8], i: &mut usize) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *b.get(*i)?;
            *i += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }
    // iterates (field number, wire type, payload) of one protobuf message
    fn fields(b: &[u8]) -> impl Iterator<Item = (u64, &[u8], u64)> {
        let mut i = 0;
        std::iter::from_fn(move || {
            if i >= b.len() {
                return None;
            }
            let key = varint(b, &mut i)?;
            let (field, wire) = (key >> 3, key & 7);
            match wire {
                0 => varint(b, &mut i).map(|v| (field, &b[0..0], v)),
                1 => {
                    i += 8;
                    Some((field, &b[0..0], 0))
                }
                2 => {
                    let n = varint(b, &mut i)? as usize;
                    let s = b.get(i..i + n)?;
                    i += n;
                    Some((field, s, 0))
                }
                5 => {
                    i += 4;
                    Some((field, &b[0..0], 0))
                }
                _ => None,
            }
        })
    }
    let (mut pieces, mut special) = (Vec::new(), Vec::new());
    for (field, data, _) in fields(buf) {
        if field != 1 {
            continue;
        }
        let (mut piece, mut kind) = (String::new(), 1);
        for (f, d, v) in fields(data) {
            match f {
                1 => piece = String::from_utf8_lossy(d).into_owned(),
                3 => kind = v,
                _ => {}
            }
        }
        // 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte
        special.push(matches!(kind, 2 | 3 | 5));
        pieces.push(piece);
    }
    if pieces.is_empty() {
        return Err("tokenizer has no pieces".into());
    }
    Ok((pieces, special))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_matches_direct_convolution() {
        let backend = Backend::new(false);
        for (c_in, c_out, k, stride, len) in [(1usize, 4usize, 10usize, 5usize, 103usize), (3, 2, 3, 2, 21), (3, 2, 2, 2, 20)] {
            let x: Vec<f32> = (0..len * c_in).map(|i| ((i * 7919) % 97) as f32 / 97.0 - 0.5).collect(); // [frames][channels]
            let w: Vec<f32> = (0..c_out * c_in * k).map(|i| ((i * 104729) % 89) as f32 / 89.0 - 0.5).collect(); // torch (out, in, k)
            let out = (len - k) / stride + 1;
            let expect: Vec<f32> = (0..out)
                .flat_map(|i| {
                    let (x, w) = (&x, &w);
                    (0..c_out).map(move |o| (0..c_in).flat_map(|ci| (0..k).map(move |j| (ci, j))).map(|(ci, j)| w[(o * c_in + ci) * k + j] * x[(i * stride + j) * c_in + ci]).sum::<f32>())
                })
                .collect();
            // weights as uploaded by 'Fix::ConvKIn', (out, k, in)
            let wk: Vec<f32> = (0..c_out).flat_map(|o| (0..k).flat_map(move |j| (0..c_in).map(move |ci| (o, j, ci)))).map(|(o, j, ci)| w[(o * c_in + ci) * k + j]).collect();
            let mut weights = Weights::new(4);
            let wt = weights.tensor(F32, &[(k * c_in) as i64, c_out as i64]);
            weights.alloc(&backend).unwrap();
            weights.set(wt, &wk);
            let got: Vec<f32> = backend
                .run(|g| {
                    let input = g.input(F32, &[c_in as i64, len as i64]);
                    (conv(g, input, wt, k as i64, stride as i64), vec![(input, ggml::f32_bytes(&x))])
                })
                .unwrap();
            let diff = got.iter().zip(&expect).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert_eq!(got.len(), expect.len());
            assert!(diff < 1e-4, "k={k} s={stride}: {diff}");
        }
    }

    #[test]
    fn ctc_decoding() {
        let pieces: Vec<String> = ["<pad>", "▁", "a", "b", "▁c", " "].iter().map(|s| s.to_string()).collect();
        let special = [true, false, false, false, false, false];
        // "aab cb" with blanks and repeats, then " a" after a literal-space piece
        let ids = [0, 2, 2, 0, 2, 3, 3, 1, 1, 0, 4, 0, 3, 5, 2];
        let w = ctc_words(&ids, &pieces, &special);
        let t: Vec<_> = w.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(t, [" aab", " cb", " a"]);
        assert!((w[0].t0 - 0.02).abs() < 1e-9 && (w[0].t1 - 0.12).abs() < 1e-9);
        assert!((w[1].t0 - 0.20).abs() < 1e-9);
    }

    #[test]
    fn sentencepiece_reader() {
        // ModelProto { pieces: [{piece:"<pad>", type:CONTROL}, {piece:"▁ä", score:0}] }
        let mut buf = Vec::new();
        for (p, kind) in [("<pad>", 3u8), ("▁ä", 1)] {
            let mut sp = vec![0x0a, p.len() as u8];
            sp.extend_from_slice(p.as_bytes());
            sp.extend_from_slice(&[0x15, 0, 0, 0, 0, 0x18, kind]);
            buf.push(0x0a);
            buf.push(sp.len() as u8);
            buf.extend(sp);
        }
        let (pieces, special) = read_sentencepiece(&buf).unwrap();
        assert_eq!(pieces, ["<pad>", "▁ä"]);
        assert_eq!(special, [true, false]);
    }
}
