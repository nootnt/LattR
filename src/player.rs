//! Audio output: reads 48 kHz PCM from disk, time-stretches with WSOLA (keeps pitch at any speed) and resamples to the device rate

use crate::audio::{PcmReader, RATE};
use crate::i18n::{tr, trf};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Ctl {
    playing: bool,
    speed: f32,
    volume: f32,
    denoised: bool,
    pos: f64,
    seek: Option<f64>,
    len: usize,
    new_readers: Option<[Option<PcmReader>; 2]>,
}

pub struct Player {
    ctl: Arc<Mutex<Ctl>>,
    _stream: cpal::Stream,
}

impl Player {
    pub fn new() -> Result<Self, String> {
        let device = cpal::default_host().default_output_device().ok_or(tr("no audio output device"))?;
        let config = device.default_output_config().map_err(|e| e.to_string())?;
        let ctl = Arc::new(Mutex::new(Ctl { speed: 1.0, volume: 1.0, ..Default::default() }));
        let fmt = config.sample_format();
        let config: cpal::StreamConfig = config.into();
        let stream = match fmt {
            cpal::SampleFormat::F32 => build::<f32>(&device, config, ctl.clone()),
            cpal::SampleFormat::I16 => build::<i16>(&device, config, ctl.clone()),
            cpal::SampleFormat::U16 => build::<u16>(&device, config, ctl.clone()),
            cpal::SampleFormat::I32 => build::<i32>(&device, config, ctl.clone()),
            f => return Err(trf("unsupported output format {0}", &[&f.to_string()])),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self { ctl, _stream: stream })
    }

    fn ctl(&self) -> std::sync::MutexGuard<'_, Ctl> {
        self.ctl.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn set_tracks(&self, orig: Option<PcmReader>, den: Option<PcmReader>) {
        let mut c = self.ctl();
        c.len = orig.as_ref().map_or(0, |r| r.len);
        c.new_readers = Some([orig, den]);
        c.playing = false;
        c.seek = Some(0.0);
        c.pos = 0.0;
    }

    pub fn set_denoised_reader(&self, den: PcmReader) {
        let mut c = self.ctl();
        let orig = c.new_readers.as_mut().and_then(|r| r[0].take());
        // Keep whichever original reader the audio thread already holds
        c.new_readers = Some([orig, Some(den)]);
    }

    pub fn playing(&self) -> bool {
        self.ctl().playing
    }

    pub fn set_playing(&self, on: bool) {
        let mut c = self.ctl();
        if on && c.pos >= c.len as f64 {
            c.seek = Some(0.0);
            c.pos = 0.0;
        }
        c.playing = on && c.len > 0;
    }

    pub fn pos(&self) -> f64 {
        let c = self.ctl();
        c.seek.unwrap_or(c.pos) / RATE as f64
    }

    pub fn seek(&self, secs: f64) {
        let mut c = self.ctl();
        let s = (secs.max(0.0) * RATE as f64).min(c.len as f64);
        c.seek = Some(s);
        c.pos = s;
    }

    pub fn set_speed(&self, s: f32) {
        self.ctl().speed = s;
    }

    pub fn set_volume(&self, v: f32) {
        self.ctl().volume = v;
    }

    pub fn set_denoised(&self, on: bool) {
        self.ctl().denoised = on;
    }
}

fn build<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: cpal::StreamConfig, ctl: Arc<Mutex<Ctl>>) -> Result<cpal::Stream, String> {
    let channels = config.channels as usize;
    let step = RATE as f64 / config.sample_rate as f64;
    let mut readers: [Option<Cache>; 2] = [None, None];
    let mut wsola = Wsola::new();
    let mut hist = [0f32; 4];
    let mut frac = 0f64;
    let mut mono = Vec::new();
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
                let frames = data.len() / channels;
                mono.clear();
                mono.resize(frames, 0f32);
                let volume = {
                    let mut c = ctl.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(r) = c.new_readers.take() {
                        let [o, d] = r;
                        if let Some(o) = o {
                            readers[0] = Some(Cache::new(o));
                        }
                        readers[1] = d.map(Cache::new);
                    }
                    if let Some(s) = c.seek.take() {
                        wsola.reset(s);
                        hist = [0.0; 4];
                    }
                    let idx = usize::from(c.denoised && readers[1].is_some());
                    if c.playing && readers[idx].is_some() {
                        wsola.speed = c.speed as f64;
                        let src = readers[idx].as_mut().unwrap();
                        for m in mono.iter_mut() {
                            while frac >= 1.0 {
                                hist.rotate_left(1);
                                hist[3] = wsola.next(src);
                                frac -= 1.0;
                            }
                            *m = cubic(&hist, frac as f32);
                            frac += step;
                        }
                        c.pos = wsola.pos();
                        if c.pos >= c.len as f64 {
                            c.playing = false;
                        }
                    }
                    c.volume
                };
                for (frame, &m) in data.chunks_mut(channels).zip(&mono) {
                    let v = T::from_sample((m * volume).clamp(-1.0, 1.0));
                    frame.iter_mut().for_each(|s| *s = v);
                }
            },
            |e| eprintln!("audio stream error: {e}"),
            None,
        )
        .map_err(|e| e.to_string())
}

/// Catmull-Rom interpolation between h[1] and h[2]
fn cubic(h: &[f32; 4], t: f32) -> f32 {
    let (a, b, c, d) = (h[0], h[1], h[2], h[3]);
    b + 0.5 * t * (c - a + t * (2.0 * a - 5.0 * b + 4.0 * c - d + t * (3.0 * (b - c) + d - a)))
}

/// One-second block cache over the PCM file so the audio thread rarely touches disk
struct Cache {
    reader: PcmReader,
    start: usize,
    block: Vec<i16>,
}

impl Cache {
    const BLOCK: usize = RATE as usize;

    fn new(reader: PcmReader) -> Self {
        Self { reader, start: usize::MAX, block: Vec::new() }
    }

    fn read(&mut self, start: i64, out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate() {
            let p = start + i as i64;
            *o = if p < 0 || p as usize >= self.reader.len {
                0.0
            } else {
                let p = p as usize;
                if p < self.start || p >= self.start + self.block.len() {
                    self.start = p / Self::BLOCK * Self::BLOCK;
                    self.block.resize(Self::BLOCK, 0);
                    let n = self.reader.read(self.start, &mut self.block);
                    self.block.truncate(n.max(1));
                }
                self.block.get(p - self.start).map_or(0.0, |&s| s as f32 / 32768.0)
            };
        }
    }
}

/// 32 ms frames, 10 ms search tolerance at 48 kHz.
const N: usize = 1536;
const HS: usize = N / 2;
const TOL: i64 = 480;

/// Waveform-similarity overlap-add time stretching.
struct Wsola {
    speed: f64,
    window: Vec<f32>,
    /// Nominal source position of the next analysis frame.
    ana: f64,
    /// Start of the previously chosen frame, None right after a seek.
    prev: Option<i64>,
    tail: Vec<f32>,
    out: Vec<f32>,
    out_pos: usize,
    /// Source position corresponding to out[0] and the step per output sample.
    out_src: f64,
    out_step: f64,
    frame: Vec<f32>,
    cand: Vec<f32>,
    target: Vec<f32>,
}

impl Wsola {
    fn new() -> Self {
        let window = (0..N).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos()).collect();
        Self {
            speed: 1.0,
            window,
            ana: 0.0,
            prev: None,
            tail: vec![0.0; HS],
            out: Vec::new(),
            out_pos: 0,
            out_src: 0.0,
            out_step: 1.0,
            frame: vec![0.0; N],
            cand: vec![0.0; N + 2 * TOL as usize],
            target: vec![0.0; N],
        }
    }

    fn reset(&mut self, pos: f64) {
        self.ana = pos;
        self.prev = None;
        self.tail.iter_mut().for_each(|x| *x = 0.0);
        self.out.clear();
        self.out_pos = 0;
        self.out_src = pos;
    }

    fn pos(&self) -> f64 {
        self.out_src + self.out_pos as f64 * self.out_step
    }

    fn next(&mut self, src: &mut Cache) -> f32 {
        if self.out_pos >= self.out.len() {
            self.hop(src);
        }
        self.out_pos += 1;
        self.out[self.out_pos - 1]
    }

    fn hop(&mut self, src: &mut Cache) {
        let nominal = self.ana.round() as i64;
        let chosen = match self.prev {
            None => nominal,
            Some(prev) => {
                src.read(prev + HS as i64, &mut self.target);
                src.read(nominal - TOL, &mut self.cand);
                let mut best = (f32::MIN, 0usize);
                for k in 0..=2 * TOL as usize {
                    let c = &self.cand[k..k + N];
                    let (mut dot, mut en) = (0f32, 1e-9f32);
                    for i in (0..N).step_by(4) {
                        dot += c[i] * self.target[i];
                        en += c[i] * c[i];
                    }
                    let score = dot / en.sqrt();
                    // ties (e.g. silence) prefer the smallest shift
                    if score > best.0 || (score == best.0 && k.abs_diff(TOL as usize) < best.1.abs_diff(TOL as usize)) {
                        best = (score, k);
                    }
                }
                nominal - TOL + best.1 as i64
            }
        };
        src.read(chosen, &mut self.frame);
        self.out.clear();
        for i in 0..HS {
            self.out.push(self.tail[i] + self.frame[i] * self.window[i]);
            self.tail[i] = self.frame[HS + i] * self.window[HS + i];
        }
        self.out_pos = 0;
        self.out_src = self.ana;
        self.out_step = self.speed;
        self.prev = Some(chosen);
        self.ana += HS as f64 * self.speed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(samples: &[i16]) -> Cache {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("lattr-wsola-{}-{n}.pcm", std::process::id()));
        std::fs::write(&path, samples.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        Cache::new(PcmReader::from_file(file, samples.len()))
    }

    fn stretch(speed: f64, len: usize) -> (Vec<f32>, Vec<f32>) {
        let input: Vec<i16> = (0..len).map(|i| ((i as f32 * 0.05).sin() * (i as f32 * 0.001).cos() * 8000.0) as i16).collect();
        let mut src = source(&input);
        let mut w = Wsola::new();
        w.reset(0.0);
        w.speed = speed;
        let mut out = Vec::new();
        while w.pos() < len as f64 {
            out.push(w.next(&mut src));
        }
        (input.iter().map(|&s| s as f32 / 32768.0).collect(), out)
    }

    #[test]
    fn identity_at_normal_speed() {
        let (input, out) = stretch(1.0, 48000);
        // the first half-frame fades in
        let err = input[HS..45000].iter().zip(&out[HS..45000]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-4, "{err}");
    }

    #[test]
    fn duration_follows_speed() {
        for speed in [0.5, 2.0, 5.0] {
            let (input, out) = stretch(speed, 96000);
            let ratio = out.len() as f64 / input.len() as f64;
            assert!((ratio - 1.0 / speed).abs() < 0.02, "speed {speed}: ratio {ratio}");
        }
    }
}
