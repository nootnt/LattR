//! Decoding (ffmpeg), noise reduction and disk-backed PCM storage
//! Audio is stored as 48 kHz mono i16 on disk (~1.4 GB per version for 4 h)

use crate::i18n::trf;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

/// Storage and playback rate
pub const RATE: u32 = 48_000;
/// Speech models take 16 kHz
pub const ASR_RATE: u32 = 16_000;
/// Samples per envelope bucket (10 ms)
pub const HOP: usize = 480;

/// Peak envelope with a max-pyramid for fast drawing at any zoom level
pub struct Envelope {
    levels: Vec<Vec<f32>>,
}

impl Envelope {
    fn new(peaks: Vec<f32>) -> Self {
        let mut levels = vec![peaks];
        while levels.last().unwrap().len() > 1 {
            let next = levels.last().unwrap().chunks(2).map(|c| c.iter().fold(0f32, |a, &b| a.max(b))).collect();
            levels.push(next);
        }
        Self { levels }
    }

    pub fn len(&self) -> usize {
        self.levels[0].len()
    }

    /// Max peak over buckets 'a..b'
    pub fn max(&self, a: usize, b: usize) -> f32 {
        let b = b.min(self.len());
        if a >= b {
            return 0.0;
        }
        let lvl = ((b - a).ilog2() as usize).saturating_sub(1).min(self.levels.len() - 1);
        let v = &self.levels[lvl];
        v[a >> lvl..=((b - 1) >> lvl).min(v.len() - 1)].iter().fold(0f32, |m, &x| m.max(x))
    }

    pub fn peaks(&self) -> &[f32] {
        &self.levels[0]
    }
}

/// A decoded track stored in a temp file, deleted on drop
pub struct Track {
    path: PathBuf,
    pub len: usize,
    pub env: Envelope,
}

impl Track {
    pub fn reader(&self) -> std::io::Result<PcmReader> {
        Ok(PcmReader { file: File::open(&self.path)?, len: self.len })
    }

    pub fn duration(&self) -> f64 {
        self.len as f64 / RATE as f64
    }
}

impl Drop for Track {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub struct PcmReader {
    file: File,
    pub len: usize,
}

impl PcmReader {
    #[cfg(test)]
    pub fn from_file(file: File, len: usize) -> Self {
        Self { file, len }
    }

    /// Reads samples starting at 'start', returns the count read
    pub fn read(&mut self, start: usize, out: &mut [i16]) -> usize {
        let n = out.len().min(self.len.saturating_sub(start));
        let mut bytes = vec![0u8; n * 2];
        if self.file.seek(SeekFrom::Start(start as u64 * 2)).is_err() || self.file.read_exact(&mut bytes).is_err() {
            return 0;
        }
        for (o, b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *o = i16::from_le_bytes(*b);
        }
        n
    }

    /// Samples 'start..start + len', zero outside the file
    fn read_f32(&mut self, start: i64, len: usize) -> Vec<f32> {
        let mut buf = vec![0i16; len];
        let skip = (-start).clamp(0, len as i64) as usize;
        self.read(start.max(0) as usize, &mut buf[skip..]);
        buf.iter().map(|&s| s as f32 / 32768.0).collect()
    }

    /// Length in 16 kHz samples
    pub fn asr_len(&self) -> usize {
        self.len / 3
    }

    /// 16 kHz samples 'start..start + len', low-passed and decimated
    pub fn read_asr(&mut self, start: usize, len: usize) -> Vec<f32> {
        let taps = decimation_taps();
        let half = taps.len() / 2;
        let x = self.read_f32((start * 3) as i64 - half as i64, len * 3 + taps.len());
        (0..len).map(|k| x[3 * k..3 * k + taps.len()].iter().zip(taps).map(|(a, b)| a * b).sum()).collect()
    }
}

pub enum LoadMsg {
    Original(Track),
    /// Denoised track plus per-10 ms voice probability from RNNoise
    Denoised(Track, Vec<f32>),
    Error(String),
}

#[derive(Default)]
pub struct LoadProgress {
    /// Progress in permille, for (original, denoised)
    pub orig: AtomicU64,
    pub den: AtomicU64,
}

fn ffmpeg_path() -> PathBuf {
    let name = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    std::env::current_exe()
        .ok()
        .and_then(|p| Some(p.parent()?.join(name)))
        .filter(|p| p.exists())
        .unwrap_or_else(|| name.into())
}

/// Decodes 'src' with ffmpeg to mono PCM at 'rate' and feeds chunks to 'sink'
/// 'progress' receives a 0..1 fraction from the reported duration
fn decode(src: &Path, rate: u32, fmt: &str, bytes_per_sample: usize, mut sink: impl FnMut(&[u8]), progress: impl Fn(f64)) -> Result<(), String> {
    let mut cmd = Command::new(ffmpeg_path());
    // local files only
    let mut input = std::ffi::OsString::from("file:");
    input.push(src);
    cmd.args(["-nostdin", "-hide_banner", "-protocol_whitelist", "file", "-i"])
        .arg(input)
        .args(["-vn", "-ac", "1", "-ar", &rate.to_string(), "-f", fmt, "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| trf("could not run ffmpeg ({0}), install ffmpeg or place it next to the program", &[&e.to_string()]))?;

    let duration = Arc::new(Mutex::new(0f64));
    let stderr = child.stderr.take().unwrap();
    let dur = duration.clone();
    let log = std::thread::spawn(move || {
        let mut tail = Vec::new();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Some(d) = line.trim().strip_prefix("Duration: ").and_then(parse_hms) {
                *dur.lock().unwrap() = d;
            }
            tail.push(line);
            if tail.len() > 5 {
                tail.remove(0);
            }
        }
        tail.join("\n")
    });

    let mut out = child.stdout.take().unwrap();
    let mut buf = vec![0u8; 1 << 16];
    let mut carry = Vec::new();
    let mut total = 0usize;
    loop {
        let n = out.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        carry.extend_from_slice(&buf[..n]);
        let whole = carry.len() / bytes_per_sample * bytes_per_sample;
        sink(&carry[..whole]);
        carry.drain(..whole);
        total += whole / bytes_per_sample;
        let d = *duration.lock().unwrap();
        if d > 0.0 {
            progress((total as f64 / rate as f64 / d).min(1.0));
        }
    }
    let status = child.wait().map_err(|e| e.to_string())?;
    let tail = log.join().unwrap_or_default();
    if !status.success() || total == 0 {
        return Err(trf("ffmpeg failed to decode the file:\n{0}", &[&tail]));
    }
    Ok(())
}

fn parse_hms(s: &str) -> Option<f64> {
    let t = s.split(',').next()?;
    let mut parts = t.split(':');
    let h: f64 = parts.next()?.parse().ok()?;
    let m: f64 = parts.next()?.parse().ok()?;
    let s: f64 = parts.next()?.parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

/// Writes i16 samples to a temp file while collecting the 10 ms peak envelope
struct TrackWriter {
    path: PathBuf,
    out: BufWriter<File>,
    len: usize,
    peaks: Vec<f32>,
    cur: f32,
}

impl TrackWriter {
    fn new(tag: &str) -> Result<Self, String> {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let path = crate::config::cache_dir().join(format!("{}-{n}-{tag}.pcm", std::process::id()));
        let file = File::create(&path).map_err(|e| trf("cannot create {0}: {1}", &[&path.display().to_string(), &e.to_string()]))?;
        Ok(Self { path, out: BufWriter::with_capacity(1 << 20, file), len: 0, peaks: Vec::new(), cur: 0.0 })
    }

    fn push(&mut self, s: i16) -> std::io::Result<()> {
        self.out.write_all(&s.to_le_bytes())?;
        self.cur = self.cur.max((s as f32 / 32768.0).abs());
        self.len += 1;
        if self.len.is_multiple_of(HOP) {
            self.peaks.push(self.cur);
            self.cur = 0.0;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Track, String> {
        self.out.flush().map_err(|e| e.to_string())?;
        if !self.len.is_multiple_of(HOP) {
            self.peaks.push(self.cur);
        }
        Ok(Track { path: self.path.clone(), len: self.len, env: Envelope::new(std::mem::take(&mut self.peaks)) })
    }
}

/// Windowed-sinc low-pass for 48 kHz -> 16 kHz
fn decimation_taps() -> &'static [f32] {
    static TAPS: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    TAPS.get_or_init(|| {
        const N: usize = 63;
        let fc = 7_200.0 / 48_000.0;
        let mut taps: Vec<f32> = (0..N)
            .map(|i| {
                let x = i as f32 - (N - 1) as f32 / 2.0;
                let sinc = if x == 0.0 { 2.0 * fc } else { (2.0 * std::f32::consts::PI * fc * x).sin() / (std::f32::consts::PI * x) };
                let w = 0.42 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (N - 1) as f32).cos()
                    + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / (N - 1) as f32).cos();
                sinc * w
            })
            .collect();
        let sum: f32 = taps.iter().sum();
        taps.iter_mut().for_each(|t| *t /= sum);
        taps
    })
}

fn to_i16(x: f32) -> i16 {
    x.clamp(-32768.0, 32767.0) as i16
}

/// Starts decoding 'src' in the background, original track first
pub fn load(src: PathBuf, tx: Sender<LoadMsg>, progress: Arc<LoadProgress>, notify: impl Fn() + Send + Clone + 'static) {
    let (src2, tx2, prog2, notify2) = (src.clone(), tx.clone(), progress.clone(), notify.clone());
    std::thread::spawn(move || {
        let res = (|| {
            let mut w = TrackWriter::new("orig")?;
            let mut err = None;
            decode(
                &src,
                RATE,
                "s16le",
                2,
                |b| {
                    for c in b.as_chunks::<2>().0 {
                        if let Err(e) = w.push(i16::from_le_bytes([c[0], c[1]])) {
                            err.get_or_insert(e.to_string());
                        }
                    }
                },
                |p| {
                    progress.orig.store((p * 1000.0) as u64, Ordering::Relaxed);
                    notify();
                },
            )?;
            if let Some(e) = err {
                return Err(e);
            }
            w.finish()
        })();
        let _ = tx.send(res.map(LoadMsg::Original).unwrap_or_else(LoadMsg::Error));
        notify();
    });

    std::thread::spawn(move || {
        let res = (|| {
            let mut w = TrackWriter::new("den")?;
            let mut den = nnnoiseless::DenoiseState::new();
            let mut frame = Vec::with_capacity(nnnoiseless::FRAME_SIZE);
            let mut out = [0f32; nnnoiseless::FRAME_SIZE];
            let mut vad = Vec::new();
            let mut err = None;
            let mut first = true;
            decode(
                &src2,
                RATE,
                "f32le",
                4,
                |b| {
                    for c in b.as_chunks::<4>().0 {
                        frame.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * 32768.0);
                        if frame.len() == nnnoiseless::FRAME_SIZE {
                            vad.push(den.process_frame(&mut out, &frame));
                            frame.clear();
                            // RNNoise fades in over its first frame, skip it
                            if std::mem::take(&mut first) {
                                continue;
                            }
                            for &s in &out {
                                if let Err(e) = w.push(to_i16(s)) {
                                    err.get_or_insert(e.to_string());
                                }
                            }
                        }
                    }
                },
                |p| {
                    prog2.den.store((p * 1000.0) as u64, Ordering::Relaxed);
                    notify2();
                },
            )?;
            if let Some(e) = err {
                return Err(e);
            }
            Ok((w.finish()?, vad))
        })();
        let _ = tx2.send(res.map(|(t, v)| LoadMsg::Denoised(t, v)).unwrap_or_else(LoadMsg::Error));
        notify2();
    });
}

/// Removes leftover PCM files from crashed or killed sessions
pub fn clean_cache() {
    let Ok(dir) = std::fs::read_dir(crate::config::cache_dir()) else { return };
    for e in dir.flatten() {
        let path = e.path();
        if path.extension().is_none_or(|x| x != "pcm") {
            continue;
        }
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|d| d.as_secs() > 86_400);
        // files are named "<pid>-...", on Linux check whether that process is gone
        let pid = path.file_name().and_then(|n| n.to_str()?.split('-').next()?.parse::<u32>().ok());
        let dead = cfg!(target_os = "linux") && pid.is_some_and(|p| p != std::process::id() && !Path::new(&format!("/proc/{p}")).exists());
        if old || dead {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Encodes 16 kHz mono samples as a WAV file in memory
pub fn wav_bytes(samples: &[f32]) -> Vec<u8> {
    let data_len = samples.len() as u32 * 2;
    let mut v = Vec::with_capacity(44 + data_len as usize);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&ASR_RATE.to_le_bytes());
    v.extend_from_slice(&(ASR_RATE * 2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        v.extend_from_slice(&to_i16(s * 32768.0).to_le_bytes());
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader(samples: &[i16]) -> PcmReader {
        let path = std::env::temp_dir().join(format!("lattr-asr-{}.pcm", std::process::id()));
        std::fs::write(&path, samples.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        let file = File::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        PcmReader::from_file(file, samples.len())
    }

    #[test]
    fn decimation_keeps_speech_band_and_timing() {
        // 1 kHz tone plus a 20 kHz tone that must not alias into the 16 kHz signal
        let tone = |f: f32, i: usize| (2.0 * std::f32::consts::PI * f * i as f32 / RATE as f32).sin();
        let input: Vec<i16> = (0..RATE as usize).map(|i| ((tone(1000.0, i) + tone(20000.0, i)) * 8000.0) as i16).collect();
        let mut r = reader(&input);
        let out = r.read_asr(1000, 8000);
        let expect = |k: usize| tone(1000.0, (1000 + k) * 3) * 8000.0 / 32768.0;
        let err = (0..8000).map(|k| (out[k] - expect(k)).abs()).fold(0f32, f32::max);
        assert!(err < 0.01, "max error {err}");
        assert_eq!(r.asr_len(), RATE as usize / 3);
        // reading past the ends pads with silence instead of failing
        assert_eq!(r.read_asr(r.asr_len() - 10, 100).len(), 100);
    }
}
