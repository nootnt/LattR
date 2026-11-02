//! Transcription: splits audio into speech chunks (skipping long silences) and runs
//! them through a local whisper.cpp model or a cloud API, giving timed words

use crate::audio::{ASR_RATE as RATE, PcmReader, wav_bytes};

/// 10 ms at 16 kHz: RNNoise voice-activity frames and envelope buckets
const HOP: usize = 160;
use crate::i18n::{tr, trf};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

#[derive(Default)]
pub struct Word {
    /// Leading whitespace marks a word boundary
    pub text: String,
    pub t0: f64,
    pub t1: f64,
    pub speaker: Option<u32>,
}

pub enum Msg {
    Progress(f32),
    Status(String),
    Words(Vec<Word>),
    /// Progress of speaker identification
    SpeakerProgress(f32),
    /// Finished speaker turns (start, end, speaker)
    Speakers(Vec<(f64, f64, u32)>),
    SpeakerError(String),
    Done,
    Error(String),
}

#[derive(Clone)]
pub enum Backend {
    Whisper { model: PathBuf, gpu: bool, accurate: bool },
    /// Meta Omnilingual ASR checkpoint
    Omni { model: PathBuf, gpu: bool },
    OpenAi { url: String, key: String, model: String },
    Google { url: String, key: String },
}

impl Backend {
    /// Longest chunk the backend accepts, in seconds
    fn max_chunk(&self) -> f64 {
        match self {
            Backend::Whisper { .. } => 300.0,
            Backend::Omni { .. } => crate::omni::MAX_SECONDS,
            Backend::OpenAi { .. } => 600.0, // 25 MB upload limit
            Backend::Google { .. } => 55.0,  // sync recognize limit is 60 s
        }
    }
}

pub struct Job {
    pub backend: Backend,
    pub language: String,
    pub reader: PcmReader,
    pub vad: Vec<f32>,
    pub peaks: Vec<f32>,
    /// Resume time after a cancelled run
    pub start: f64,
    pub speakers: Option<SpeakerJob>,
    /// Omnilingual checkpoint for re-timing, plus GPU flag
    pub align: Option<(PathBuf, bool)>,
}

pub struct SpeakerJob {
    pub model: PathBuf,
    pub reader: PcmReader,
    /// 'None' estimates
    pub count: Option<usize>,
    pub gpu: bool,
}

struct Piece {
    src: usize,
    len: usize,
    at: usize,
}

struct Chunk {
    pieces: Vec<Piece>,
}

/// Silence inserted between joined speech regions
const GAP: usize = RATE as usize / 2;

impl Chunk {
    fn samples(&self, r: &mut PcmReader) -> Vec<f32> {
        let mut out = Vec::new();
        for p in &self.pieces {
            out.resize(p.at, 0.0);
            out.extend(r.read_asr(p.src, p.len));
        }
        out
    }

    /// Maps a time inside the chunk buffer back to the source timeline
    fn map(&self, t: f64) -> f64 {
        let s = (t.max(0.0) * RATE as f64) as usize;
        let i = self.pieces.partition_point(|p| p.at <= s).saturating_sub(1);
        let p = &self.pieces[i];
        (p.src + (s.saturating_sub(p.at)).min(p.len)) as f64 / RATE as f64
    }
}

/// Speech regions in 10 ms frames from RNNoise voice probabilities
/// 'pad' frames are added around detected speech and gaps up to 'merge' frames are bridged
pub(crate) fn speech_regions(vad: &[f32], pad: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut regions: Vec<(usize, usize)> = Vec::new();
    for (i, _) in vad.iter().enumerate().filter(|(_, v)| **v > 0.35) {
        let (a, b) = (i.saturating_sub(pad), (i + 1 + pad).min(vad.len()));
        match regions.last_mut() {
            Some(last) if a <= last.1 + merge => last.1 = b,
            _ => regions.push((a, b)),
        }
    }
    regions
}

/// Group speech regions into chunks no longer than 'max' seconds
fn plan(vad: &[f32], peaks: &[f32], len: usize, start: usize, max: f64) -> Vec<Chunk> {
    let max = (max * RATE as f64) as usize;
    let frame = HOP; // RNNoise frames and envelope buckets are both 10 ms
    // 0.5 s pad, gaps under 2 s stay in a region
    let mut regions: Vec<(usize, usize)> = speech_regions(vad, 50, 200)
        .into_iter()
        .map(|(a, b)| ((a * frame).max(start), (b * frame).min(len)))
        .filter(|(a, b)| a < b)
        .collect();
    if vad.is_empty() {
        regions = vec![(start, len)];
    }
    // split overlong regions at the quietest point near the limit
    let mut split = Vec::new();
    for (mut a, b) in regions {
        while b - a > max {
            let lo = (a + max * 3 / 4) / frame;
            let hi = (a + max) / frame;
            let cut = (lo..hi).min_by(|&x, &y| peaks.get(x).unwrap_or(&0.0).total_cmp(peaks.get(y).unwrap_or(&0.0))).unwrap_or(hi) * frame;
            split.push((a, cut));
            a = cut;
        }
        split.push((a, b));
    }
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cur = Chunk { pieces: Vec::new() };
    let mut used = 0;
    for (a, b) in split {
        let need = if cur.pieces.is_empty() { b - a } else { GAP + b - a };
        if used + need > max && !cur.pieces.is_empty() {
            chunks.push(std::mem::replace(&mut cur, Chunk { pieces: Vec::new() }));
            used = 0;
        }
        let at = if cur.pieces.is_empty() { 0 } else { used + GAP };
        cur.pieces.push(Piece { src: a, len: b - a, at });
        used = at + b - a;
    }
    if !cur.pieces.is_empty() {
        chunks.push(cur);
    }
    chunks
}

pub fn spawn(mut job: Job, tx: Sender<Msg>, cancel: Arc<AtomicBool>, notify: Arc<dyn Fn() + Send + Sync>) {
    std::thread::spawn(move || {
        let send = |m: Msg| {
            let _ = tx.send(m);
            notify();
        };
        if let Some(sj) = job.speakers.take() {
            let (tx, notify, cancel, vad) = (tx.clone(), notify.clone(), cancel.clone(), job.vad.clone());
            std::thread::spawn(move || {
                let send = |m: Msg| {
                    let _ = tx.send(m);
                    notify();
                };
                let mut reader = sj.reader;
                let res = crate::speaker::Embedder::load(&sj.model, sj.gpu).and_then(|m| {
                    crate::speaker::diarize(&m, &mut reader, &vad, sj.count, &mut |p| {
                        send(Msg::SpeakerProgress(p));
                        !cancel.load(Ordering::Relaxed)
                    })
                });
                send(res.map_or_else(Msg::SpeakerError, Msg::Speakers));
            });
        }
        let start = (job.start * RATE as f64) as usize;
        let chunks = plan(&job.vad, &job.peaks, job.reader.asr_len(), start, job.backend.max_chunk());
        let total: usize = chunks.iter().flat_map(|c| &c.pieces).map(|p| p.len).sum::<usize>().max(1);
        let mut done = 0usize;
        let mut local = None;
        let mut omni = None;
        // load the alignment model while transcription starts
        let mut aligner_load = job.align.clone().filter(|_| !matches!(job.backend, Backend::Omni { .. })).map(|(model, gpu)| {
            std::thread::spawn(move || crate::omni::Omni::load(&model, gpu).map(crate::align::Aligner::new))
        });
        let mut aligner = None;
        for chunk in &chunks {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let samples = chunk.samples(&mut job.reader);
            let chunk_len: usize = chunk.pieces.iter().map(|p| p.len).sum();
            let base = done as f32 / total as f32;
            let frac = chunk_len as f32 / total as f32;
            let res = match &job.backend {
                Backend::Whisper { model, gpu, accurate } => {
                    if local.is_none() {
                        send(Msg::Status(tr("Loading model...").into()));
                        match Local::new(model, *gpu) {
                            Ok(l) => local = Some(l),
                            Err(e) => return send(Msg::Error(e)),
                        }
                        send(Msg::Status(tr("Transcribing...").into()));
                    }
                    let (tx, notify) = (tx.clone(), notify.clone());
                    let prog = move |p: i32| {
                        let _ = tx.send(Msg::Progress(base + frac * p as f32 / 100.0));
                        notify();
                    };
                    local.as_mut().unwrap().run(&samples, &job.language, *accurate, &cancel, prog)
                }
                Backend::Omni { model, gpu } => {
                    if omni.is_none() {
                        send(Msg::Status(tr("Loading model…").into()));
                        match crate::omni::Omni::load(model, *gpu) {
                            Ok(m) => omni = Some(m),
                            Err(e) => return send(Msg::Error(e)),
                        }
                        let gpu = omni.as_ref().is_some_and(|m| m.on_gpu());
                        send(Msg::Status(if gpu { tr("Transcribing on GPU…") } else { tr("Transcribing…") }.into()));
                    }
                    omni.as_ref().unwrap().transcribe(&samples)
                }
                Backend::OpenAi { url, key, model } => openai(url, key, model, &job.language, &samples),
                Backend::Google { url, key } => google(url, key, &job.language, &samples),
            };
            match res {
                Ok(mut words) => {
                    if let Some(h) = aligner_load.take() {
                        match h.join() {
                            Ok(Ok(a)) => aligner = Some(a),
                            Ok(Err(e)) => send(Msg::Status(trf("Word timing refinement failed: {0}", &[&e]))),
                            Err(_) => send(Msg::Status(trf("Word timing refinement failed: {0}", &["crashed"]))),
                        }
                    }
                    if let Some(a) = &aligner
                        && let Err(e) = a.align(&samples, &mut words)
                    {
                        send(Msg::Status(trf("Word timing refinement failed: {0}", &[&e])));
                    }
                    let words = words.into_iter().map(|w| Word { t0: chunk.map(w.t0), t1: chunk.map(w.t1), ..w }).collect();
                    send(Msg::Words(words));
                }
                Err(_) if cancel.load(Ordering::Relaxed) => break,
                Err(e) => return send(Msg::Error(trf("at {0}: {1}", &[&crate::app::fmt_time(chunk.pieces[0].src as f64 / RATE as f64, false), &e]))),
            }
            done += chunk_len;
            send(Msg::Progress(done as f32 / total as f32));
        }
        send(Msg::Done);
    });
}

struct Local {
    ctx: whisper_rs::WhisperContext,
    dtw: bool,
}

fn dtw_preset(model: &std::path::Path) -> Option<whisper_rs::DtwModelPreset> {
    use whisper_rs::DtwModelPreset::*;
    let name = model.file_name()?.to_string_lossy().to_lowercase();
    let name = name.trim_start_matches("ggml-");
    let presets = [
        ("large-v3-turbo", LargeV3Turbo),
        ("large-v3", LargeV3),
        ("large-v2", LargeV2),
        ("large-v1", LargeV1),
        ("medium.en", MediumEn),
        ("medium", Medium),
        ("small.en", SmallEn),
        ("small", Small),
        ("base.en", BaseEn),
        ("base", Base),
        ("tiny.en", TinyEn),
        ("tiny", Tiny),
    ];
    presets.into_iter().find(|(p, _)| name.starts_with(p)).map(|(_, m)| m)
}

impl Local {
    fn new(model: &std::path::Path, gpu: bool) -> Result<Self, String> {
        if std::env::var_os("LATTR_DEBUG").is_none() {
            whisper_rs::install_logging_hooks();
        }
        let mut params = whisper_rs::WhisperContextParameters::default();
        params.use_gpu(gpu);
        let preset = dtw_preset(model);
        if let Some(p) = preset.clone() {
            params.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset { model_preset: p };
        }
        // Windows fails on non-ASCII model paths
        #[cfg(windows)]
        let data = std::fs::read(model).map_err(|e| trf("failed to load model: {0}", &[&e.to_string()]))?;
        #[cfg(not(windows))]
        let path = model.to_str().ok_or(tr("model path is not valid UTF-8"))?;
        let ctx = {
            let _init = crate::ggml::INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            #[cfg(windows)]
            let ctx = whisper_rs::WhisperContext::new_from_buffer_with_params(&data, params);
            #[cfg(not(windows))]
            let ctx = whisper_rs::WhisperContext::new_with_params(path, params);
            ctx.map_err(|e| trf("failed to load model: {0}", &[&e.to_string()]))?
        };
        Ok(Self { ctx, dtw: preset.is_some() })
    }

    fn run(&mut self, samples: &[f32], language: &str, accurate: bool, cancel: &Arc<AtomicBool>, progress: impl FnMut(i32) + 'static) -> Result<Vec<Word>, String> {
        use whisper_rs::{FullParams, SamplingStrategy};
        // creating a state sets up whisper.cpp's GPU backend
        let mut state = {
            let _init = crate::ggml::INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            self.ctx.create_state().map_err(|e| e.to_string())?
        };
        let strategy = if accurate { SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 } } else { SamplingStrategy::Greedy { best_of: 5 } };
        let mut p = FullParams::new(strategy);
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
        p.set_n_threads(threads as i32);
        p.set_language(Some(language));
        p.set_translate(false);
        p.set_token_timestamps(true);
        p.set_suppress_blank(true);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_special(false);
        p.set_print_timestamps(false);
        let c = cancel.clone();
        // whisper-rs stores 'Box<dyn FnMut>' but casts callback data to 'F', so pass a boxed dyn closure
        let abort: Box<dyn FnMut() -> bool> = Box::new(move || c.load(Ordering::Relaxed));
        p.set_abort_callback_safe(abort);
        p.set_progress_callback_safe(progress);
        state.full(p, samples).map_err(|e| e.to_string())?;
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }

        let eot = self.ctx.token_eot();
        let mut tokens: Vec<(Vec<u8>, f64, f64)> = Vec::new();
        for seg in state.as_iter() {
            let toks: Vec<(&[u8], whisper_rs::WhisperTokenData)> = (0..seg.n_tokens())
                .filter_map(|i| seg.get_token(i))
                .filter(|t| t.token_id() < eot)
                .filter_map(|t| Some((t.to_bytes().ok()?, t.token_data())))
                .collect();
            let use_dtw = self.dtw && dtw_plausible(&toks, seg.start_timestamp(), seg.end_timestamp());
            for (tb, d) in toks {
                let t0 = if use_dtw { d.t_dtw } else { d.t0 } as f64 / 100.0;
                tokens.push((tb.to_vec(), t0, (d.t1 as f64 / 100.0).max(t0)));
            }
        }
        Ok(group_tokens(&tokens))
    }
}

/// Joins subword tokens (raw bytes, possibly splitting a UTF-8 character) into words
fn group_tokens(tokens: &[(impl AsRef<[u8]>, f64, f64)]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut bytes: Vec<u8> = Vec::new();
    let flush = |bytes: &mut Vec<u8>, words: &mut Vec<Word>| {
        if let Some(w) = words.last_mut()
            && !bytes.is_empty()
        {
            w.text = String::from_utf8_lossy(bytes).into_owned();
        }
        bytes.clear();
    };
    // a token with an inner space holds two words, split it before the space
    let pieces = tokens.iter().flat_map(|(tb, t0, t1)| {
        let tb = tb.as_ref();
        let cuts: Vec<usize> = (1..tb.len()).filter(|&i| tb[i] == b' ' && tb[i - 1] != b' ').collect();
        let bounds: Vec<usize> = std::iter::once(0).chain(cuts).chain(std::iter::once(tb.len())).collect();
        bounds.windows(2).map(move |w| (&tb[w[0]..w[1]], *t0, *t1)).collect::<Vec<_>>()
    });
    for (tb, t0, t1) in pieces {
        if starts_word(&bytes, tb) {
            flush(&mut bytes, &mut words);
            words.push(Word { t0, t1, ..Default::default() });
        }
        bytes.extend_from_slice(tb);
        if let Some(w) = words.last_mut() {
            w.t1 = w.t1.max(t1);
        }
    }
    flush(&mut bytes, &mut words);
    // keep word ends from overlapping the next word's start
    for i in 1..words.len() {
        let next = words[i].t0;
        let w = &mut words[i - 1];
        if w.t1 > next {
            w.t1 = next.max(w.t0);
        }
    }
    words.retain(|w| !w.text.trim().is_empty());
    words
}

/// Trusts DTW onsets only when they stay in the segment and keep advancing
fn dtw_plausible(toks: &[(&[u8], whisper_rs::WhisperTokenData)], seg_t0: i64, seg_t1: i64) -> bool {
    let mut stuck = 0;
    for w in toks.windows(2) {
        let (a, b) = (w[0].1.t_dtw, w[1].1.t_dtw);
        if b < a {
            return false;
        }
        stuck = if b == a { stuck + 1 } else { 0 };
        if stuck >= 3 {
            return false;
        }
    }
    toks.iter().all(|(_, d)| d.t_dtw >= seg_t0 - 50 && d.t_dtw <= seg_t1 + 50)
}

/// Scripts written without spaces, where each token is treated as its own word
fn spaceless(c: char) -> bool {
    matches!(c as u32,
        0x0E00..=0x0EFF | 0x1000..=0x109F | 0x1780..=0x17FF | 0x3040..=0x30FF |
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F)
}

fn starts_word(cur: &[u8], tok: &[u8]) -> bool {
    if cur.is_empty() || tok.first() == Some(&b' ') {
        return true;
    }
    let (Ok(c), Ok(t)) = (std::str::from_utf8(cur), std::str::from_utf8(tok)) else { return false };
    let (last, first) = (c.chars().last(), t.chars().next());
    first.is_some_and(spaceless) || (last.is_some_and(spaceless) && first.is_some_and(char::is_alphanumeric))
}

fn agent() -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .http_status_as_error(false)
            .timeout_global(Some(std::time::Duration::from_secs(600)))
            .build(),
    )
}

fn read_json(resp: Result<ureq::http::Response<ureq::Body>, ureq::Error>) -> Result<serde_json::Value, String> {
    let mut resp = resp.map_err(|e| e.to_string())?;
    let status = resp.status();
    // long recordings give large word lists
    let body = resp.body_mut().with_config().limit(256 << 20).read_to_string().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("HTTP {status}: {}", body.chars().take(500).collect::<String>()));
    }
    serde_json::from_str(&body).map_err(|e| format!("bad response: {e}"))
}

/// OpenAI-compatible '/audio/transcriptions' endpoint
fn openai(url: &str, key: &str, model: &str, language: &str, samples: &[f32]) -> Result<Vec<Word>, String> {
    let boundary = format!("----lattr{:x}", std::process::id() as u64 ^ samples.len() as u64);
    let mut body = Vec::new();
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
    };
    field("model", model);
    field("response_format", "verbose_json");
    field("timestamp_granularities[]", "word");
    field("timestamp_granularities[]", "segment");
    if language != "auto" {
        field("language", language);
    }
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    body.extend_from_slice(&wav_bytes(samples));
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let endpoint = format!("{}/audio/transcriptions", url.trim_end_matches('/'));
    let local = ["http://localhost", "http://127.0.0.1", "http://[::1]"].iter().any(|p| endpoint.starts_with(p));
    if !endpoint.starts_with("https://") && !local {
        return Err(tr("the API URL must start with https:// (plain http only for localhost)").into());
    }
    let resp = agent()
        .post(&endpoint)
        .header("Authorization", &format!("Bearer {key}"))
        .header("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
        .send(&body[..]);
    let v = read_json(resp)?;
    let text = v["text"].as_str().unwrap_or_default();
    let raw: Vec<Word> = v["words"]
        .as_array()
        .ok_or(tr("response has no word timestamps (the model may not support them)"))?
        .iter()
        .map(|w| Word {
            text: w["word"].as_str().unwrap_or_default().trim().to_string(),
            t0: w["start"].as_f64().unwrap_or(0.0),
            t1: w["end"].as_f64().unwrap_or(0.0),
            speaker: None,
        })
        .collect();
    Ok(restore_punctuation(text, raw))
}

fn restore_punctuation(text: &str, words: Vec<Word>) -> Vec<Word> {
    let norm = |c: char| c.to_lowercase().next().unwrap_or(c);
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut pos = 0;
    let mut out = Vec::with_capacity(words.len());
    for w in words {
        let target: Vec<char> = w.text.chars().filter(|c| c.is_alphanumeric()).map(norm).collect();
        let mut found = None;
        'search: for s in pos..chars.len().min(pos + 64) {
            if target.is_empty() || norm(chars[s].1) != target[0] {
                continue;
            }
            let (mut i, mut k) = (s, 0);
            while i < chars.len() && k < target.len() {
                let c = chars[i].1;
                if c.is_alphanumeric() {
                    if norm(c) != target[k] {
                        continue 'search;
                    }
                    k += 1;
                } else if c.is_whitespace() {
                    continue 'search;
                }
                i += 1;
            }
            if k == target.len() {
                found = Some(i);
                break;
            }
        }
        match found {
            Some(mut end) => {
                while end < chars.len() && !chars[end].1.is_whitespace() && !chars[end].1.is_alphanumeric() {
                    end += 1;
                }
                let b0 = chars[pos].0;
                let b1 = chars.get(end).map_or(text.len(), |c| c.0);
                let piece = &text[b0..b1];
                let lead = if piece.starts_with(char::is_whitespace) || out.is_empty() { " " } else { "" };
                out.push(Word { text: format!("{lead}{}", piece.trim()), ..w });
                pos = end;
            }
            None => out.push(Word { text: format!(" {}", w.text), ..w }),
        }
    }
    out
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            s.push(if i <= c.len() { T[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    s
}

/// Accepts Google Cloud or a compatible base URL
fn google_endpoint(base: &str) -> String {
    let base = base.trim().trim_end_matches('/');
    let base = if base.is_empty() { crate::config::GOOGLE_URL } else { base };
    format!("{base}/speech:recognize")
}

/// Google Cloud Speech-to-Text v1 synchronous recognition (API key auth).
fn google(url: &str, key: &str, language: &str, samples: &[f32]) -> Result<Vec<Word>, String> {
    if language == "auto" {
        return Err(tr("Google needs a language; pick one instead of Auto").into());
    }
    let pcm: Vec<u8> = wav_bytes(samples)[44..].to_vec();
    let req = serde_json::json!({
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": RATE,
            "languageCode": language,
            "enableWordTimeOffsets": true,
            "enableAutomaticPunctuation": true,
            "model": "latest_long",
        },
        "audio": { "content": base64(&pcm) },
    });
    let endpoint = google_endpoint(url);
    let local = ["http://localhost", "http://127.0.0.1", "http://[::1]"].iter().any(|p| endpoint.starts_with(p));
    if !endpoint.starts_with("https://") && !local {
        return Err(tr("the API URL must start with https:// (plain http only for localhost)").into());
    }
    // keep the key out of URLs and logs
    let resp = agent()
        .post(&endpoint)
        .header("X-goog-api-key", key)
        .header("Content-Type", "application/json")
        .send(req.to_string().as_bytes());
    let v = read_json(resp)?;
    let secs = |s: &serde_json::Value| s.as_str().and_then(|s| s.trim_end_matches('s').parse::<f64>().ok()).unwrap_or(0.0);
    let spaced = !["ja", "zh", "th", "lo", "km", "my"].iter().any(|l| language.starts_with(l));
    let mut words = Vec::new();
    for r in v["results"].as_array().into_iter().flatten() {
        for w in r["alternatives"][0]["words"].as_array().into_iter().flatten() {
            let text = w["word"].as_str().unwrap_or_default();
            words.push(Word { text: if spaced { format!(" {text}") } else { text.into() }, t0: secs(&w["startTime"]), t1: secs(&w["endTime"]), speaker: None });
        }
    }
    Ok(words)
}

pub fn languages() -> Vec<(&'static str, String)> {
    (0..=whisper_rs::get_lang_max_id())
        .filter_map(|i| {
            let code = whisper_rs::get_lang_str(i)?;
            let name = whisper_rs::get_lang_str_full(i)?;
            let mut c = name.chars();
            let name = c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default();
            Some((code, name))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64() {
        assert_eq!(base64(b"hello"), "aGVsbG8=");
        assert_eq!(base64(b"hi!"), "aGkh");
        assert_eq!(base64(b"a"), "YQ==");
    }

    #[test]
    fn punctuation() {
        let w = |t: &str| Word { text: t.into(), ..Default::default() };
        let out = restore_punctuation(" Hello, world! It's me.", vec![w("hello"), w("world"), w("its"), w("me")]);
        let t: Vec<_> = out.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(t, [" Hello,", " world!", " It's", " me."]);
    }

    #[test]
    fn regions() {
        let mut vad = vec![0.0; 3000];
        vad[100..200].iter_mut().for_each(|v| *v = 0.9);
        vad[250..300].iter_mut().for_each(|v| *v = 0.9);
        vad[2000..2100].iter_mut().for_each(|v| *v = 0.9);
        assert_eq!(speech_regions(&vad, 50, 200), vec![(50, 350), (1950, 2150)]);
        let chunks = plan(&vad, &vec![0.0; 3000], 3000 * HOP, 0, 55.0);
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.pieces.len(), 2);
        // a time in the second piece maps back past the skipped silence
        let t_second = (c.pieces[1].at as f64 / RATE as f64) + 0.5;
        assert!((c.map(t_second) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn google_endpoint_defaults_and_accepts_compatible_hosts() {
        assert_eq!(google_endpoint(""), "https://speech.googleapis.com/v1/speech:recognize");
        assert_eq!(google_endpoint("https://speech.googleapis.com/v1/"), "https://speech.googleapis.com/v1/speech:recognize");
        assert_eq!(google_endpoint(" https://example.com/speech/v1 "), "https://example.com/speech/v1/speech:recognize");
    }

    /// Splits 'text' into byte tokens of the given sizes, like BPE does, ignoring char boundaries
    fn byte_tokens(text: &str, sizes: &[usize]) -> Vec<(Vec<u8>, f64, f64)> {
        let b = text.as_bytes();
        let mut out = Vec::new();
        let (mut at, mut i) = (0, 0);
        while at < b.len() {
            let n = sizes[i % sizes.len()].min(b.len() - at);
            out.push((b[at..at + n].to_vec(), i as f64, i as f64 + 0.5));
            at += n;
            i += 1;
        }
        out
    }

    fn texts(words: &[Word]) -> Vec<&str> {
        words.iter().map(|w| w.text.as_str()).collect()
    }

    #[test]
    fn group_german() {
        // "ü" (C3 BC) and "ß" (C3 9F) split across tokens
        let toks: Vec<(&[u8], f64, f64)> = vec![
            (b" Ich", 0.0, 0.3),
            (b" hei\xC3", 0.4, 0.6),
            (b"\x9Fe", 0.6, 0.8),
            (b" J\xC3", 1.0, 1.2),
            (b"\xBCrgen", 1.2, 1.5),
            (b".", 1.5, 1.6),
        ];
        let w = group_tokens(&toks);
        assert_eq!(texts(&w), [" Ich", " heiße", " Jürgen."]);
        assert_eq!((w[1].t0, w[1].t1), (0.4, 0.8));
        assert_eq!((w[2].t0, w[2].t1), (1.0, 1.6));
    }

    #[test]
    fn group_german_split_tokens() {
        let text = " Grüße, Welt! Wie geht's?";
        for sizes in [&[1][..], &[2], &[4], &[5, 2], &[7, 3, 1]] {
            let w = group_tokens(&byte_tokens(text, sizes));
            assert_eq!(texts(&w), [" Grüße,", " Welt!", " Wie", " geht's?"], "sizes {sizes:?}");
            assert!(w.windows(2).all(|p| p[0].t0 < p[1].t0 && p[0].t1 <= p[1].t0));
        }
    }

    #[test]
    fn whisper_languages_include_german() {
        assert!(languages().iter().any(|l| l.0 == "de"));
    }

    #[test]
    fn group_spaceless_scripts() {
        let toks: Vec<(&[u8], f64, f64)> = vec![
            ("こん".as_bytes(), 0.0, 0.2),
            ("にちは".as_bytes(), 0.2, 0.5),
            ("。".as_bytes(), 0.5, 0.6),
            ("世界".as_bytes(), 1.0, 1.3),
        ];
        assert_eq!(texts(&group_tokens(&toks)), ["こん", "にちは。", "世界"]);
        // mixed Latin inside Japanese still separates
        let toks: Vec<(&[u8], f64, f64)> = vec![("東京".as_bytes(), 0.0, 0.2), (b" Tokyo", 0.3, 0.6), ("です".as_bytes(), 0.7, 0.9)];
        assert_eq!(texts(&group_tokens(&toks)), ["東京", " Tokyo", "です"]);
    }

    #[test]
    fn punctuation_german() {
        let w = |t: &str| Word { text: t.into(), ..Default::default() };
        let out = restore_punctuation(
            " „Guten Morgen“, sagte die Straßenbahnfahrerin. Grüß Gott!",
            ["guten", "morgen", "sagte", "die", "straßenbahnfahrerin", "grüß", "gott"].map(w).into(),
        );
        assert_eq!(texts(&out), [" „Guten", " Morgen“,", " sagte", " die", " Straßenbahnfahrerin.", " Grüß", " Gott!"]);
    }

    #[test]
    fn punctuation_unmatched_word_is_kept() {
        let w = |t: &str| Word { text: t.into(), ..Default::default() };
        let out = restore_punctuation(" Das ist gut.", ["das", "ist", "sehr", "gut"].map(w).into());
        assert_eq!(texts(&out), [" Das", " ist", " sehr", " gut."]);
    }

    /// End-to-end run: 'LATTR_AUDIO=file LATTR_MODEL=ggml.bin cargo test --release e2e -- --ignored --nocapture'
    #[test]
    #[ignore]
    fn e2e() {
        use crate::audio::{LoadMsg, LoadProgress, load};
        let audio = std::env::var("LATTR_AUDIO").unwrap();
        let model = std::env::var("LATTR_MODEL").unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::time::Instant::now();
        load(audio.into(), tx, Arc::new(LoadProgress::default()), || {});
        let (mut orig, mut den) = (None, None);
        while orig.is_none() || den.is_none() {
            match rx.recv().unwrap() {
                LoadMsg::Original(o) => {
                    println!("original decoded in {:?}", t.elapsed());
                    orig = Some(o)
                }
                LoadMsg::Denoised(d, v) => {
                    println!("denoised in {:?}", t.elapsed());
                    den = Some((d, v))
                }
                LoadMsg::Error(e) => panic!("{e}"),
            }
        }
        let (orig, (den, vad)) = (orig.unwrap(), den.unwrap());
        println!("len {} / {} samples, vad frames {}", orig.len, den.len, vad.len());
        let job = Job {
            backend: if std::env::var_os("LATTR_OMNI").is_some() {
                Backend::Omni { model: model.into(), gpu: std::env::var("LATTR_GPU").is_ok() }
            } else {
                Backend::Whisper { model: model.into(), gpu: std::env::var("LATTR_GPU").is_ok(), accurate: std::env::var("LATTR_ACCURATE").is_ok() }
            },
            language: std::env::var("LATTR_LANG").unwrap_or("auto".into()),
            reader: orig.reader().unwrap(),
            peaks: orig.env.peaks().to_vec(),
            vad,
            start: 0.0,
            align: std::env::var("LATTR_ALIGN").ok().map(|m| (m.into(), std::env::var_os("LATTR_GPU").is_some())),
            speakers: std::env::var("LATTR_SPK_MODEL").ok().map(|m| SpeakerJob { model: m.into(), reader: orig.reader().unwrap(), count: None, gpu: std::env::var_os("LATTR_GPU").is_some() }),
        };
        let chunks = plan(&job.vad, &job.peaks, job.reader.asr_len(), 0, 300.0);
        for c in &chunks {
            let p: Vec<_> = c.pieces.iter().map(|p| (p.src as f64 / 16000.0, p.len as f64 / 16000.0)).collect();
            println!("chunk {:?}", p);
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::time::Instant::now();
        spawn(job, tx, Arc::new(AtomicBool::new(false)), Arc::new(|| {}));
        let mut words = Vec::new();
        let mut timed = Vec::new();
        let (mut done, mut turns) = (false, if std::env::var_os("LATTR_SPK_MODEL").is_some() { None } else { Some(Vec::new()) });
        for m in rx {
            match m {
                Msg::Words(w) => w.into_iter().for_each(|w| {
                    println!("{:8.2} {:8.2} {:?} {:?}", w.t0, w.t1, w.text, w.speaker);
                    timed.push((w.t0, w.t1, w.text.clone()));
                    words.push(w.text);
                }),
                Msg::Speakers(found) => {
                    println!("speakers identified after {:?}", t.elapsed());
                    turns = Some(found)
                }
                Msg::Error(e) | Msg::SpeakerError(e) => panic!("{e}"),
                Msg::Done => {
                    println!("text done after {:?}", t.elapsed());
                    done = true
                }
                _ => {}
            }
            if done && turns.is_some() {
                break;
            }
        }
        for (a, b, w) in &timed {
            if let Some(s) = crate::speaker::speaker_at(turns.as_deref().unwrap_or_default(), (a + b) / 2.0) {
                println!("spk {a:8.2} {b:8.2} {w:?} {s}");
            }
        }
        println!("transcribed in {:?}", t.elapsed());
        if let Ok(out) = std::env::var("LATTR_OUT") {
            std::fs::write(out, words.concat()).unwrap();
        }
        if let Ok(expect) = std::env::var("LATTR_EXPECT") {
            assert!(words.iter().any(|w| w.contains(&expect)), "expected {expect:?} in {words:?}");
        }
    }
}
