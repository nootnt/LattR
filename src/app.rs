use crate::audio::{self, LoadMsg, LoadProgress, Track};
use crate::config::{self, BackendKind, Config, MODELS, Mode};
use crate::i18n::{self, tr, trf};
use crate::player::Player;
use crate::transcribe::{self, Backend, Msg};
use crate::transcript::Transcript;
use crate::waveform::{self, View};
use egui::text::CCursor;
use egui::{Align, Color32, Key, Modifiers, RichText};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PRE_ROLL: f64 = 0.25;

const AUDIO_EXTS: &[&str] = &["mp3", "m4a", "wav", "flac", "ogg", "opus", "aac", "wma", "webm", "mp4", "mkv", "mov", "avi"];

struct Audio {
    path: PathBuf,
    orig: Option<Arc<Track>>,
    den: Option<Arc<Track>>,
    vad: Vec<f32>,
    rx: Receiver<LoadMsg>,
    progress: Arc<LoadProgress>,
}

struct Job {
    rx: Receiver<Msg>,
    cancel: Arc<AtomicBool>,
    progress: f32,
    status: String,
    started: Instant,
    /// Transcription start time
    from: f64,
    /// Progress of speaker identification
    speakers: Option<f32>,
    turns: Option<Vec<(f64, f64, u32)>>,
    text_done: bool,
}

struct Download {
    stem: &'static str,
    done: Arc<AtomicU64>,
    total: Arc<AtomicU64>,
    cancel: Arc<AtomicBool>,
    result: Arc<Mutex<Option<Result<(), String>>>>,
}

pub struct App {
    cfg: Config,
    player: Result<Player, String>,
    audio: Option<Audio>,
    transcript: Transcript,
    project: Option<PathBuf>,
    dirty: Option<Instant>,
    job: Option<Job>,
    view: View,
    denoised: bool,
    last_cursor: Option<usize>,
    last_word: Option<usize>,
    last_row_y: f32,
    status: String,
    show_settings: bool,
    download: Option<Download>,
    queue: Vec<&'static str>,
    model_prompt: bool,
    speaker_names: Vec<String>,
    languages: Vec<(&'static str, String)>,
}

const MIN_SPEED: f32 = 0.1;
const MAX_SPEED: f32 = 5.0;

/// Slider position (0..1) for a playback speed
fn speed_to_pos(speed: f32) -> f64 {
    let s = speed.clamp(MIN_SPEED, MAX_SPEED);
    let p = if s <= 1.0 { 0.5 * (s / MIN_SPEED).ln() / (1.0 / MIN_SPEED).ln() } else { 0.5 + 0.5 * s.ln() / MAX_SPEED.ln() };
    p as f64
}

fn pos_to_speed(pos: f64) -> f32 {
    let p = pos.clamp(0.0, 1.0) as f32;
    if p <= 0.5 { MIN_SPEED * (1.0 / MIN_SPEED).powf(p * 2.0) } else { MAX_SPEED.powf((p - 0.5) * 2.0) }
}

/// Speed slides in 0.05 steps
fn snap_speed(speed: f32) -> f32 {
    ((speed / 0.05).round() * 0.05).clamp(MIN_SPEED, MAX_SPEED)
}

fn engine_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Whisper => tr("Whisper (local)"),
        BackendKind::Omni => tr("Omnilingual (local)"),
        BackendKind::OpenAi => tr("OpenAI-compatible API"),
        BackendKind::Google => tr("Google Cloud API"),
    }
}

fn engine_hint(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Whisper => tr("Runs on this computer - best for common languages"),
        BackendKind::Omni => tr("Omnilingual ASR on this computer - many languages. Lowercase, no punctuation."),
        BackendKind::OpenAi => tr("Online - OpenAI, Groq or a self-hosted server"),
        BackendKind::Google => tr("Online - Google Cloud Speech-to-Text"),
    }
}

/// Shortens 'max' chars in the middle with "…"
fn ellipsize_middle(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    let tail = (max - 1) / 2;
    let head = max - 1 - tail;
    let start: String = s.chars().take(head).collect();
    let end: String = s.chars().skip(n - tail).collect();
    format!("{}…{}", start.trim_end(), end.trim_start())
}

pub fn fmt_time(t: f64, ms: bool) -> String {
    let t = t.max(0.0);
    let (h, m, s) = ((t / 3600.0) as u64, (t / 60.0) as u64 % 60, t % 60.0);
    match (h > 0, ms) {
        (true, true) => format!("{h}:{m:02}:{s:04.1}"),
        (true, false) => format!("{h}:{m:02}:{:02}", s as u64),
        (false, true) => format!("{m}:{s:04.1}"),
        (false, false) => format!("{m}:{:02}", s as u64),
    }
}

/// Transcript next to the audio, or in data dir if that's not writable
fn project_paths(audio: &Path) -> [PathBuf; 2] {
    let mut name = audio.as_os_str().to_owned();
    name.push(".transcript.json");
    // stable across runs
    let hash = audio.to_string_lossy().bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    let dir = config::models_dir().with_file_name("projects");
    let _ = std::fs::create_dir_all(&dir);
    [PathBuf::from(name), dir.join(format!("{hash:016x}.json"))]
}

impl App {
    pub fn new() -> Self {
        let cfg = Config::load();
        i18n::set(&cfg.ui_language);
        let has_model = std::fs::read_dir(config::models_dir()).into_iter().flatten().flatten().any(|e| e.path().extension().is_some_and(|x| x == "bin"))
            || [&cfg.model_fast, &cfg.model_accurate].into_iter().flatten().any(|m| m.exists());
        let model_prompt = !has_model && cfg.backend == BackendKind::Whisper;
        let player = Player::new();
        if let Ok(p) = &player {
            p.set_speed(cfg.speed);
            p.set_volume(cfg.volume);
        }
        let app = Self {
            cfg,
            player,
            audio: None,
            transcript: Transcript::default(),
            project: None,
            dirty: None,
            job: None,
            view: View { start: 0.0, len: 60.0 },
            denoised: false,
            last_cursor: None,
            last_word: None,
            last_row_y: f32::NAN,
            status: String::new(),
            show_settings: false,
            download: None,
            queue: Vec::new(),
            model_prompt,
            speaker_names: Vec::new(),
            languages: transcribe::languages(),
        };
        app
    }

    fn duration(&self) -> f64 {
        self.audio.as_ref().and_then(|a| a.orig.as_ref()).map_or(0.0, |t| t.duration())
    }

    fn player(&self) -> Option<&Player> {
        self.player.as_ref().ok()
    }

    fn pos(&self) -> f64 {
        self.player().map_or(0.0, |p| p.pos())
    }

    fn seek(&mut self, t: f64) {
        if let Some(p) = self.player() {
            p.seek(t);
        }
        let d = self.duration();
        if t < self.view.start || t > self.view.start + self.view.len {
            self.view.start = t - self.view.len * 0.1;
            self.view.clamp(d);
        }
    }

    pub fn open(&mut self, path: PathBuf, ctx: &egui::Context) {
        self.save();
        self.cancel_job();
        if let Some(p) = self.player() {
            p.set_tracks(None, None);
        }
        let (tx, rx) = channel();
        let progress = Arc::new(LoadProgress::default());
        let c = ctx.clone();
        audio::load(path.clone(), tx, progress.clone(), move || c.request_repaint());
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!("{name} — LattR")));

        self.transcript = Transcript::default();
        self.project = None;
        for p in project_paths(&path) {
            if let Some(mut t) = std::fs::read(&p).ok().and_then(|b| serde_json::from_slice::<Transcript>(&b).ok()) {
                t.sanitize();
                self.transcript = t;
                self.project = Some(p);
                break;
            }
        }
        self.status = if self.project.is_some() { tr("Loaded saved transcript").into() } else { String::new() };
        self.audio = Some(Audio { path, orig: None, den: None, vad: Vec::new(), rx, progress });
        self.dirty = None;
        self.last_cursor = None;
        self.last_word = None;
        self.view = View { start: 0.0, len: 60.0 };
    }

    fn save(&mut self) {
        let Some(a) = &self.audio else { return };
        if self.dirty.take().is_none() {
            return;
        }
        let Ok(json) = serde_json::to_vec(&self.transcript) else { return };
        let paths = project_paths(&a.path);
        let order = if self.project.as_ref() == Some(&paths[1]) { [&paths[1], &paths[0]] } else { [&paths[0], &paths[1]] };
        for p in order {
            let tmp = p.with_extension("json.tmp");
            if std::fs::write(&tmp, &json).and_then(|_| std::fs::rename(&tmp, p)).is_ok() {
                self.project = Some(p.clone());
                return;
            }
            let _ = std::fs::remove_file(&tmp);
        }
        self.status = tr("Could not save transcript").into();
    }

    fn poll(&mut self) {
        if let Some(a) = &mut self.audio {
            while let Ok(m) = a.rx.try_recv() {
                match m {
                    LoadMsg::Original(t) => {
                        let t = Arc::new(t);
                        if let Ok(p) = &self.player {
                            p.set_tracks(t.reader().ok(), a.den.as_ref().and_then(|d| d.reader().ok()));
                        }
                        self.view = View { start: 0.0, len: t.duration().min(60.0) };
                        a.orig = Some(t);
                    }
                    LoadMsg::Denoised(t, vad) => {
                        let t = Arc::new(t);
                        if let (Ok(p), Some(_), Ok(r)) = (&self.player, &a.orig, t.reader()) {
                            p.set_denoised_reader(r);
                        }
                        a.den = Some(t);
                        a.vad = vad;
                    }
                    LoadMsg::Error(e) => self.status = e,
                }
            }
        }
        let mut finished = false;
        if let Some(j) = &mut self.job {
            while let Ok(m) = j.rx.try_recv() {
                match m {
                    Msg::Progress(p) => j.progress = p,
                    Msg::Status(s) => j.status = s,
                    Msg::Words(words) => {
                        for w in words {
                            let speaker = w.speaker.or_else(|| j.turns.as_ref().and_then(|t| crate::speaker::speaker_at(t, (w.t0 + w.t1) / 2.0)));
                            self.transcript.push_word(&w.text, w.t0, w.t1, speaker);
                        }
                        self.dirty.get_or_insert_with(Instant::now);
                    }
                    Msg::SpeakerProgress(p) => j.speakers = Some(p),
                    Msg::Speakers(turns) => {
                        // label finished words, later ones as they arrive
                        let from = j.from;
                        self.transcript.apply_speakers(from, |a, b| crate::speaker::speaker_at(&turns, (a + b) / 2.0));
                        j.turns = Some(turns);
                        j.speakers = None;
                        self.dirty.get_or_insert_with(Instant::now);
                    }
                    Msg::SpeakerError(e) => {
                        j.speakers = None;
                        if !j.cancel.load(Ordering::Relaxed) {
                            self.status = trf("Speaker detection failed: {0}", &[&e]);
                        }
                    }
                    Msg::Done => {
                        let cancelled = j.cancel.load(Ordering::Relaxed);
                        let took = fmt_time(j.started.elapsed().as_secs_f64(), false);
                        self.status = if cancelled { trf("Transcription stopped after {0}", &[&took]) } else { trf("Transcription finished after {0}", &[&took]) };
                        j.text_done = true;
                    }
                    Msg::Error(e) => {
                        self.status = trf("Transcription failed {0}", &[&e]);
                        j.cancel.store(true, Ordering::Relaxed);
                        finished = true;
                    }
                }
            }
            // speaker identification can outlast transcription
            finished |= j.text_done && j.speakers.is_none();
        }
        if finished {
            self.job = None;
            self.dirty.get_or_insert_with(Instant::now);
            self.save();
        }
        if self.dirty.is_some_and(|d| d.elapsed() > Duration::from_secs(3)) {
            self.save();
        }
        let finished = self.download.as_ref().map(|d| d.result.lock().unwrap().is_some()).unwrap_or(false);
        if finished {
            let d = self.download.take().unwrap();
            let res = d.result.lock().unwrap().take().unwrap();
            match res {
                Ok(()) => {
                    // fill the model slot unless the user already picked one
                    let slot = match config::OMNI_MODELS.iter().find(|m| m.0 == d.stem) {
                        Some(m) => self.cfg.omni_model_mut(m.2),
                        None if d.stem.starts_with("wespeaker") => &mut self.cfg.speaker_model,
                        None => self.cfg.model_mut(MODELS.iter().find(|m| m.0 == d.stem).map_or(Mode::Fast, |m| m.2)),
                    };
                    if !slot.as_ref().is_some_and(|m| m.exists()) {
                        *slot = Some(config::model_path(d.stem));
                    }
                    self.cfg.save();
                    self.status = trf("Downloaded model {0}", &[d.stem]);
                    if !self.queue.is_empty() {
                        let next = self.queue.remove(0);
                        self.start_download(next);
                    }
                }
                Err(e) => {
                    self.status = trf("Download failed: {0}", &[&e]);
                    self.queue.clear();
                }
            }
        }
    }

    fn backend(&self) -> Result<Backend, String> {
        let env = |k: &str| std::env::var(k).unwrap_or_default();
        Ok(match self.cfg.backend {
            BackendKind::Whisper => {
                let mode = self.cfg.mode;
                let missing = match mode {
                    Mode::Fast => tr("Pick/Download a fast-mode model in Settings first"),
                    Mode::Accurate => tr("Pick/Download an accurate-mode model in Settings first"),
                };
                let model = self.cfg.model(mode).clone().filter(|m| m.exists()).ok_or(missing)?;
                Backend::Whisper { model, gpu: self.cfg.use_gpu, accurate: mode == Mode::Accurate }
            }
            BackendKind::Omni => {
                            let missing = match self.cfg.mode {
                                Mode::Fast => tr("Pick/Download a fast-mode Omnilingual model in Settings first"),
                                Mode::Accurate => tr("Pick/Download an accurate-mode Omnilingual model in Settings first"),
                            };
                            let model = self.cfg.omni_model(self.cfg.mode).ok_or(missing)?;
                            Backend::Omni { model, gpu: self.cfg.use_gpu }
            }
            BackendKind::OpenAi => {
                let key = if self.cfg.openai_key.is_empty() { env("OPENAI_API_KEY") } else { self.cfg.openai_key.clone() };
                Backend::OpenAi { url: self.cfg.openai_url.clone(), key, model: self.cfg.openai_model.clone() }
            }
            BackendKind::Google => {
                let key = if self.cfg.google_key.is_empty() { env("GOOGLE_API_KEY") } else { self.cfg.google_key.clone() };
                if key.is_empty() {
                    return Err(tr("Set a Google API key in Settings").into());
                }
                Backend::Google { url: self.cfg.google_url.clone(), key }
            }
        })
    }

    fn start_job(&mut self, from: f64, ctx: &egui::Context) {
        let backend = match self.backend() {
            Ok(b) => b,
            Err(e) => {
                self.status = e;
                self.show_settings = true;
                return;
            }
        };
        let speakers = if self.cfg.detect_speakers {
            match config::speaker_model(&self.cfg.speaker_model) {
                Some(model) => Some(transcribe::SpeakerJob { model, reader: match self.audio.as_ref().and_then(|a| a.orig.as_ref()).map(|t| t.reader()) {
                    Some(Ok(r)) => r,
                    _ => return,
                }, count: (self.cfg.num_speakers > 0).then_some(self.cfg.num_speakers), gpu: self.cfg.use_gpu }),
                None => {
                    self.status = tr("Download a speaker model in Settings first, or turn speaker detection off").into();
                    self.show_settings = true;
                    return;
                }
            }
        } else {
            None
        };
        let Some(a) = &self.audio else { return };
        let (Some(orig), Some(den)) = (&a.orig, &a.den) else { return };
        let track = if self.cfg.transcribe_denoised { den } else { orig };
        let Ok(reader) = track.reader() else { return };
        if from == 0.0 {
            self.transcript = Transcript::default();
            self.last_word = None;
        }
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let has_speakers = speakers.is_some();
        // Omnilingual already has accurate word times
        let align = (self.cfg.refine_timing && matches!(self.cfg.backend, BackendKind::Whisper | BackendKind::OpenAi | BackendKind::Google))
            .then(|| self.cfg.omni_model(Mode::Fast).or_else(|| self.cfg.omni_model(Mode::Accurate)))
            .flatten()
            .map(|m| (m, self.cfg.use_gpu));
        let job = transcribe::Job { backend, language: self.cfg.language.clone(), reader, vad: a.vad.clone(), peaks: track.env.peaks().to_vec(), start: from, speakers, align };
        let c = ctx.clone();
        transcribe::spawn(job, tx, cancel.clone(), Arc::new(move || c.request_repaint()));
        let speakers = has_speakers.then_some(0.0);
        self.job = Some(Job { rx, cancel, progress: 0.0, status: tr("Transcribing...").into(), started: Instant::now(), from, speakers, turns: None, text_done: false });
    }

    fn cancel_job(&mut self) {
        if let Some(j) = &self.job {
            j.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn pick_file(&mut self, ctx: &egui::Context) {
        let file = rfd::FileDialog::new().add_filter(tr("Audio / video"), AUDIO_EXTS).add_filter(tr("All files"), &["*"]).pick_file();
        if let Some(f) = file {
            self.open(f, ctx);
        }
    }

    fn export(&mut self, kind: &str) {
        let Some(a) = &self.audio else { return };
        let stem = a.path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        let Some(path) = rfd::FileDialog::new().set_file_name(format!("{stem}.{kind}")).add_filter(kind, &[kind]).save_file() else { return };
        let data = match kind {
            "srt" => self.transcript.to_srt(),
            "json" => serde_json::to_string_pretty(&self.transcript).unwrap_or_default(),
            _ => self.transcript.text.clone(),
        };
        self.status = match std::fs::write(&path, data) {
            Ok(()) => trf("Exported {0}", &[&path.display().to_string()]),
            Err(e) => trf("Export failed: {0}", &[&e.to_string()]),
        };
    }

    fn start_download(&mut self, stem: &'static str) {
        let d = Download {
            stem,
            done: Default::default(),
            total: Default::default(),
            cancel: Default::default(),
            result: Default::default(),
        };
        let (done, total, cancel, result) = (d.done.clone(), d.total.clone(), d.cancel.clone(), d.result.clone()); std::thread::spawn(move || {
            let res = (|| -> Result<(), String> {
                for (url, dest, sha) in config::model_files(stem) {
                    if dest.exists() {
                        continue;
                    }
                    if let Some(dir) = dest.parent() {
                        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                    }
                    let part = dest.with_extension("part");
                    let mut resp = ureq::get(&url).call().map_err(|e| e.to_string())?; total.fetch_add(resp.body().content_length().unwrap_or(0), Ordering::Relaxed);
                    let mut reader = resp.body_mut().as_reader();
                    let mut file = std::fs::File::create(&part).map_err(|e| e.to_string())?;
                    let mut hash = ring::digest::Context::new(&ring::digest::SHA256);
                    let mut buf = vec![0u8; 1 << 16];
                    loop {
                        if cancel.load(Ordering::Relaxed) {
                            drop(file);
                            let _ = std::fs::remove_file(&part);
                            return Err(tr("cancelled").into());
                        }
                        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
                        if n == 0 {
                            break;
                        }
                        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
                        hash.update(&buf[..n]);
                        done.fetch_add(n as u64, Ordering::Relaxed);
                    }
                    file.sync_all().map_err(|e| e.to_string())?;
                    drop(file);
                    let got: String = hash.finish().as_ref().iter().map(|b| format!("{b:02x}")).collect();
                    if got != sha {
                        let _ = std::fs::remove_file(&part);
                        return Err(trf("{0} is damaged or was modified (checksum mismatch)", &[&url]));
                    }
                    std::fs::rename(&part, &dest).map_err(|e| e.to_string())?;
                }
                Ok(())
            })();
            *result.lock().unwrap() = Some(res);
        });
        self.download = Some(d);
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        let mut seek_by = 0.0;
        let mut speed_by = 0.0;
        let mut toggle = false;
        let mut open = false;
        let no_focus = ctx.memory(|m| m.focused().is_none());
        ctx.input_mut(|i| {
            toggle |= i.consume_key(Modifiers::NONE, Key::Escape);
            toggle |= no_focus && i.consume_key(Modifiers::NONE, Key::Space);
            if i.consume_key(Modifiers::NONE, Key::F1) {
                seek_by -= 3.0;
            }
            if i.consume_key(Modifiers::NONE, Key::F2) {
                seek_by += 3.0;
            }
            if i.consume_key(Modifiers::NONE, Key::F3) {
                speed_by -= 0.1;
            }
            if i.consume_key(Modifiers::NONE, Key::F4) {
                speed_by += 0.1;
            }
            if i.consume_key(Modifiers::COMMAND, Key::S) {
                self.dirty.get_or_insert_with(Instant::now);
            }
            open = i.consume_key(Modifiers::COMMAND, Key::O);
        });
        if open {
            self.pick_file(ctx);
        }
        if toggle {
            self.toggle_play();
        }
        if seek_by != 0.0 {
            self.seek(self.pos() + seek_by);
        }
        if speed_by != 0.0 {
            self.cfg.speed = snap_speed(self.cfg.speed + speed_by);
            if let Some(p) = self.player() {
                p.set_speed(self.cfg.speed);
            }
        }
    }

    fn toggle_play(&mut self) {
        if let Some(p) = self.player() {
            p.set_playing(!p.playing());
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                if ui.button(tr("Settings")).clicked() {
                    self.show_settings = !self.show_settings;
                }
                ui.add_enabled_ui(self.audio.is_some(), |ui| {
                    ui.menu_button(tr("Export"), |ui| {
                        for (kind, label) in [("txt", tr("Plain text (.txt)")), ("srt", tr("Subtitles (.srt)")), ("json", tr("Text + timestamps (.json)"))] {
                            if ui.button(label).clicked() {
                                self.export(kind);
                            }
                        }
                    });
                    if ui.button(tr("Save as .txt")).on_hover_text(tr("Export the transcript as a plain text file")).clicked() {
                        self.export("txt");
                    }
                });
                ui.separator();
                ui.with_layout(egui::Layout::left_to_right(Align::Center), |ui| self.toolbar_controls(ui));
            });
        });
    }

    fn toolbar_controls(&mut self, ui: &mut egui::Ui) {
        if ui.button(tr("Open...")).on_hover_text(tr("Open an audio/video file (Ctrl+O), or drop one on the window")).clicked() {
            self.pick_file(ui.ctx());
        }
        ui.separator();

        egui::ComboBox::from_id_salt("backend").selected_text(engine_name(self.cfg.backend)).show_ui(ui, |ui| {
            for kind in [BackendKind::Whisper, BackendKind::Omni, BackendKind::OpenAi, BackendKind::Google] {
                ui.selectable_value(&mut self.cfg.backend, kind, engine_name(kind)).on_hover_text(engine_hint(kind));
            }
        });
        let file = |m: Option<PathBuf>| m.as_deref().and_then(|m| m.file_name()).map_or(tr("no model chosen").into(), |n| n.to_string_lossy().into_owned());
        match self.cfg.backend {
            BackendKind::Whisper => {
                let (fast_tip, acc_tip) = (file(self.cfg.model(Mode::Fast).clone()), file(self.cfg.model(Mode::Accurate).clone()));
                ui.selectable_value(&mut self.cfg.mode, Mode::Fast, tr("Fast")).on_hover_text(trf("Greedy decoding - {0}", &[&fast_tip]));
                ui.selectable_value(&mut self.cfg.mode, Mode::Accurate, tr("Accurate"))
                    .on_hover_text(trf("Beam search, larger model - {0}\nMuch slower, best for difficult audio and rarer languages", &[&acc_tip]));
            }
            BackendKind::Omni => {
                let (fast_tip, acc_tip) = (file(self.cfg.omni_model(Mode::Fast)), file(self.cfg.omni_model(Mode::Accurate)));
                ui.selectable_value(&mut self.cfg.mode, Mode::Fast, tr("Fast")).on_hover_text(trf("Smaller model - {0}", &[&fast_tip]));
                ui.selectable_value(&mut self.cfg.mode, Mode::Accurate, tr("Accurate")).on_hover_text(trf("Big model, slower and much more accurate - {0}", &[&acc_tip]));
            }
            _ => {}
        }
        let lang_name = self.languages.iter().find(|l| l.0 == self.cfg.language).map_or(tr("Auto-detect"), |l| l.1.as_str()).to_owned();
        egui::ComboBox::from_id_salt("lang").selected_text(lang_name).height(400.0).show_ui(ui, |ui| {
            ui.selectable_value(&mut self.cfg.language, "auto".into(), tr("Auto-detect"));
            for (code, name) in &self.languages {
                ui.selectable_value(&mut self.cfg.language, code.to_string(), format!("{name} ({code})"));
            }
        });

        let ready = self.audio.as_ref().is_some_and(|a| a.orig.is_some() && a.den.is_some());
        if let Some(j) = &self.job {
            if ui.button(tr("Stop")).clicked() {
                j.cancel.store(true, Ordering::Relaxed);
            }
            let (progress, text) = match j.speakers {
                // only speaker identification is left
                Some(p) if j.text_done => (p, trf("Identifying speakers… {0}%", &[&format!("{:.0}", p * 100.0)])),
                _ => {
                    let eta = (j.progress > 0.01).then(|| j.started.elapsed().as_secs_f64() * (1.0 - j.progress as f64) / j.progress as f64);
                    let mut text = match eta {
                        Some(e) => trf("{0} {1}% · ~{2} left", &[&j.status, &format!("{:.0}", j.progress * 100.0), &fmt_time(e, false)]),
                        None => format!("{} {:.0}%", j.status, j.progress * 100.0),
                    };
                    if let Some(p) = j.speakers {
                        text.push_str(&trf(" · speakers {0}%", &[&format!("{:.0}", p * 100.0)]));
                    }
                    (j.progress, text)
                }
            };
            let room = ui.available_width() - 140.0 - if self.transcript.speakers.is_empty() { 0.0 } else { 130.0 };
            ui.add(egui::ProgressBar::new(progress).desired_width(room.clamp(120.0, 340.0)).text(text));
        } else if self.transcript.spans.is_empty() {
            let b = ui.add_enabled(ready, egui::Button::new(tr("▶ Transcribe")));
            if b.on_disabled_hover_text(tr("Waiting for the audio to finish processing")).clicked() {
                self.start_job(0.0, ui.ctx());
            }
        } else {
            ui.add_enabled_ui(ready, |ui| {
                ui.menu_button(tr("▶ Transcribe"), |ui| {
                    let last = self.transcript.spans.last().map_or(0.0, |s| s.t1);
                    if ui.button(trf("Continue after {0}", &[&fmt_time(last, false)])).clicked() {
                        self.start_job(last + 0.01, ui.ctx());
                    }
                    if ui.button(tr("Start over (replaces text)")).clicked() {
                        self.start_job(0.0, ui.ctx());
                    }
                });
            });
        }
        let label = |n: usize| if n == 0 { tr("Speakers: auto").to_string() } else { trf("Speakers: {0}", &[&n.to_string()]) };
        let selected = if self.cfg.detect_speakers { label(self.cfg.num_speakers) } else { tr("Speakers: off").into() };
        egui::ComboBox::from_id_salt("detect_speakers").selected_text(selected).show_ui(ui, |ui| {
            ui.selectable_value(&mut self.cfg.detect_speakers, false, tr("Off"));
            for n in 0..=8 {
                if n == 1 {
                    continue;
                }
                let on = self.cfg.detect_speakers && self.cfg.num_speakers == n;
                if ui.selectable_label(on, label(n)).clicked() {
                    self.cfg.detect_speakers = true;
                    self.cfg.num_speakers = n;
                }
            }
        })
        .response
        .on_hover_text(tr("Label who speaks when, using a local speaker model (applies to the next transcription)"));
        if !self.transcript.speakers.is_empty() {
            self.speakers_menu(ui);
        }
        if let Some(a) = &self.audio {
            // the file name gets leftover space
            let name = a.path.file_name().unwrap_or_default().to_string_lossy();
            let font = egui::TextStyle::Body.resolve(ui.style());
            let char_w = ui.fonts_mut(|f| f.glyph_width(&font, 'n')).max(1.0);
            let max = ((ui.available_width() / char_w) as usize).saturating_sub(2);
            if max >= 12 {
                ui.separator();
                ui.add(egui::Label::new(RichText::new(ellipsize_middle(&name, max))
                	.strong())
                	.wrap_mode(egui::TextWrapMode::Extend))
                    .on_hover_text(a.path.display().to_string());
            }
        }
    }

    fn waveforms(&mut self, ui: &mut egui::Ui) {
        let duration = self.duration();
        let pos = self.pos();
        let Some(a) = &self.audio else {
            waveform::placeholder(ui, 44.0 + 110.0 + ui.spacing().item_spacing.y, tr("Open or drop an audio file to begin"));
            return;
        };
        let Some(orig) = a.orig.clone() else {
            let p = a.progress.orig.load(Ordering::Relaxed) as f32 / 1000.0;
            waveform::placeholder(ui, 44.0 + 110.0 + ui.spacing().item_spacing.y, &trf("Decoding... {0}%", &[&format!("{:.0}", p * 100.0)]));
            return;
        };
        let den = a.den.clone();
        let use_den = self.denoised && den.is_some();
        let track = if use_den { den.as_ref().unwrap() } else { &orig };
        let color = if use_den { Color32::from_rgb(90, 190, 120) } else { Color32::from_rgb(90, 150, 230) };

        if let Some(t) = waveform::overview(ui, &track.env, duration, pos, &mut self.view, color) {
            self.seek(t);
        }
        // word labels when zoomed
        let (a_t, b_t) = (self.view.start - 5.0, self.view.start + self.view.len);
        let spans = &self.transcript.spans;
        let first = spans.partition_point(|s| s.t0 < a_t);
        let last = first + spans[first..].partition_point(|s| s.t0 <= b_t).min(400);
        let words: Vec<(f64, &str)> = self.transcript.words(first..last).into_iter().filter(|(s, _)| s.end > s.start).map(|(s, w)| (s.t0, w)).collect();
        if let Some(t) = waveform::detail(ui, &track.env, duration, pos, &mut self.view, color, &words) {
            self.seek(t);
        }
    }

    fn transport(&mut self, ui: &mut egui::Ui) {
        let duration = self.duration();
        let pos = self.pos();
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().slider_width = 110.0;
            let playing = self.player().is_some_and(|p| p.playing());
            if ui.button(tr("−3s")).on_hover_text(tr("Back 3 s (F1)")).clicked() {
                self.seek(pos - 3.0);
            }
            if ui.button(if playing { tr("⏸ Pause") } else { tr("▶ Play") }).on_hover_text(tr("Esc, or Space (when text isn't focused)")).clicked() {
                self.toggle_play();
            }
            if ui.button(tr("+3s")).on_hover_text(tr("Forward 3 s (F2)")).clicked() {
                self.seek(pos + 3.0);
            }
            ui.label(RichText::new(format!("{} / {}", fmt_time(pos, true), fmt_time(duration, false))).monospace());
            ui.separator();

            let den_ready = self.audio.as_ref().is_some_and(|a| a.den.is_some());
            ui.selectable_value(&mut self.denoised, false, tr("Original"));
            let label = match &self.audio {
                Some(a) if !den_ready && a.orig.is_some() => trf("Voice isolated ({0}%)", &[&format!("{:.0}", a.progress.den.load(Ordering::Relaxed) as f32 / 10.0)]),
                _ => tr("Voice isolated").into(),
            };
            ui.add_enabled_ui(den_ready, |ui| {
                ui.selectable_value(&mut self.denoised, true, label).on_hover_text(tr("Noise-reduced version (RNNoise)"));
            });
            if let Some(p) = self.player() {
                p.set_denoised(self.denoised);
            }
            ui.separator();

            ui.label(tr("Speed"));
            let mut pos = speed_to_pos(self.cfg.speed);
            let r = ui.add(
                egui::Slider::new(&mut pos, 0.0..=1.0)
                    .custom_formatter(|p, _| format!("{:.2}×", snap_speed(pos_to_speed(p))))
                    .custom_parser(|t| t.trim()
                    .trim_end_matches(['×', 'x'])
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .map(|v| speed_to_pos(v as f32))),
            );
            if r.changed() {
                self.cfg.speed = snap_speed(pos_to_speed(pos));
            }
            if r.double_clicked() {
                self.cfg.speed = 1.0;
            }
            ui.label(tr("Volume"));
            let mut vol = self.cfg.volume * 100.0;
            let r = ui.add(egui::Slider::new(&mut vol, 0.0..=200.0).fixed_decimals(0).suffix("%"));
            if r.changed() {
                // dragging snaps to 50% and 100%
                if r.dragged() && let Some(&m) = [50.0, 100.0].iter().find(|&&m| (vol - m).abs() < 5.0){
                    vol = m;
                }
                self.cfg.volume = vol / 100.0;
            }
            if let Some(p) = self.player() {
                p.set_speed(self.cfg.speed);
                p.set_volume(self.cfg.volume);
            }
            ui.separator();
            if ui.button("+").on_hover_text(tr("Zoom in")).clicked() {
                self.zoom(0.5, pos);
            }
            if ui.button("−").on_hover_text(tr("Zoom out")).clicked() {
                self.zoom(2.0, pos);
            }
            if ui.button(tr("Fit")).clicked() {
                self.view = View { start: 0.0, len: duration };
            }
            ui.checkbox(&mut self.cfg.follow, tr("Follow")).on_hover_text(tr("Follow current word and playhead"));
        });
        if let Err(e) = &self.player {
            ui.colored_label(ui.visuals().error_fg_color, trf("Audio output unavailable: {0}", &[e]));
        }
    }

    fn zoom(&mut self, factor: f64, around: f64) {
        let d = self.duration();
        self.view.len *= factor;
        self.view.start = around - self.view.len / 2.0;
        self.view.clamp(d);
    }

    fn editor(&mut self, ui: &mut egui::Ui) {
        let playing = self.player().is_some_and(|p| p.playing());
        let pos = self.pos();
        let word = self.transcript.span_at_time(pos);
        let font = egui::FontId::proportional(self.cfg.font_size);
        let hint = if self.audio.is_some() { tr("The transcript appears here. Click any word to jump to it.") } else { "" };
        egui::ScrollArea::vertical().id_salt("transcript").auto_shrink([false; 2]).show(ui, |ui| {
            let out = egui::TextEdit::multiline(&mut self.transcript)
                .font(font)
                .desired_width(f32::INFINITY)
                .hint_text(hint)
                .lock_focus(true)
                .frame(egui::Frame::NONE)
                .show(ui);
            if out.response.changed() {
                self.dirty.get_or_insert_with(Instant::now);
            }

            if let Some(i) = word {
                let s = self.transcript.spans[i];
                let a = out.galley.pos_from_cursor(CCursor::new(s.start));
                let mut b = out.galley.pos_from_cursor(CCursor::new(s.end));
                if b.top() > a.top() {
                    b = egui::Rect::from_min_max(egui::pos2(out.galley.rect.right(), a.top()), egui::pos2(out.galley.rect.right(), a.bottom()));
                }
                let rect = a.union(b).translate(out.galley_pos.to_vec2()).expand2(egui::vec2(2.0, 0.0));
                let hl = ui.visuals().selection.bg_fill;
                ui.painter().rect_filled(rect, 3.0, hl.gamma_multiply(0.45));
                ui.painter().hline(rect.x_range(), rect.bottom(), egui::Stroke::new(2.0, hl));
                if playing && self.cfg.follow && self.last_word != Some(i) && rect.top() != self.last_row_y {
                    ui.scroll_to_rect(rect, Some(Align::Center));
                    self.last_row_y = rect.top();
                }
            }
            self.last_word = word;

            if let Some(cr) = out.cursor_range {
                let c = cr.primary.index.0;
                if self.last_cursor != Some(c) && !out.response.changed() && cr.primary == cr.secondary
                    && let Some(t) = self.transcript.span_at_char(c).map(|s| s.t0) {
                        self.seek((t - PRE_ROLL).max(0.0));
                    }
                self.last_cursor = Some(c);
            } else {
                self.last_cursor = None;
            }
        });
    }

    fn download_row(&mut self, ui: &mut egui::Ui, stem: &'static str, desc: &'static str) {
        ui.horizontal(|ui| {
            let exists = config::model_path(stem).exists();
            match &self.download {
                Some(d) if d.stem == stem => {
                    let (done, total) = (d.done.load(Ordering::Relaxed), d.total.load(Ordering::Relaxed).max(1));
                    ui.add(egui::ProgressBar::new(done as f32 / total as f32).desired_width(160.0).show_percentage());
                    if ui.button(tr("Cancel")).clicked() {
                        d.cancel.store(true, Ordering::Relaxed);
                    }
                }
                _ => {
                    let btn = ui.add_enabled(!exists && self.download.is_none(), egui::Button::new(if exists { tr("Downloaded") } else { tr("Download") }));
                    if btn.clicked() {
                        self.start_download(stem);
                    }
                }
            }
            ui.label(tr(desc));
        });
    }

    fn speakers_menu(&mut self, ui: &mut egui::Ui) {
        let names = &self.transcript.speakers;
        if self.speaker_names.len() != names.len() {
            self.speaker_names = names.clone();
        }
        ui.menu_button(tr("Rename speakers"), |ui| {
            ui.label(tr("Rename speaker everywhere in the transcript:"));
            for i in 0..self.speaker_names.len() {
                let r = ui.text_edit_singleline(&mut self.speaker_names[i]);
                if r.lost_focus() && self.speaker_names[i] != self.transcript.speakers[i] {
                    self.transcript.rename_speaker(i, &self.speaker_names[i].clone());
                    self.speaker_names[i] = self.transcript.speakers[i].clone();
                    self.dirty.get_or_insert_with(Instant::now);
                }
            }
        });
    }

    fn model_prompt(&mut self, ctx: &egui::Context) {
        let size = |stem: &str| MODELS.iter().find(|m| m.0 == stem).map_or("", |m| tr(m.1));
        let (fast, accurate) = ("large-v3-turbo-q5_0", "large-v3-q5_0");
        let resp = egui::Modal::new(egui::Id::unique("model_prompt")).show(ctx, |ui| {
            ui.set_max_width(440.0);
            ui.heading(tr("Download a transcription model?"));
            ui.label(tr("No local model was found. Models run on this computer and are downloaded once."));
            ui.add_space(6.0);
            ui.label(trf("Fast: {0}", &[size(fast)]));
            ui.label(trf("Accurate: {0}", &[size(accurate)]));
            ui.add_space(6.0);
            let mut close = false;
            ui.horizontal(|ui| {
                if ui.button(tr("Download Fast")).clicked() {
                    self.start_download(fast);
                    close = true;
                }
                if ui.button(tr("Download both")).clicked() {
                    self.start_download(fast);
                    self.queue.push(accurate);
                    close = true;
                }
                if ui.button(tr("Not now")).clicked() {
                    close = true;
                }
            });
            ui.small(tr("You can also pick other models or use an online API in Settings."));
            close
        });
        if resp.inner || resp.should_close() {
            self.model_prompt = false;
        }
    }

    fn settings(&mut self, ctx: &egui::Context) {
        let mut open = self.show_settings;
        egui::Window::new(tr("Settings")).id(egui::Id::unique("settings")).open(&mut open).resizable(false).vscroll(true).default_width(460.0).show(ctx, |ui| {
            ui.heading(tr("Local model (whisper.cpp)"));
            let mut models: Vec<PathBuf> = std::fs::read_dir(config::models_dir())
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "bin"))
                .collect();
            for m in [&self.cfg.model_fast, &self.cfg.model_accurate].into_iter().flatten() {
                if !models.contains(m) {
                    models.push(m.clone());
                }
            }
            let name = |p: &Path| p.file_name().unwrap_or_default().to_string_lossy().into_owned();
            egui::Grid::new("models").num_columns(3).show(ui, |ui| {
                for (mode, label) in [(Mode::Fast, tr("Fast mode")), (Mode::Accurate, tr("Accurate mode"))] {
                    ui.label(label);
                    let slot = self.cfg.model_mut(mode);
                    let selected = slot.as_deref().map_or(tr("None").into(), name);
                    egui::ComboBox::from_id_salt(label).selected_text(selected).width(260.0).show_ui(ui, |ui| {
                        for m in &models {
                            ui.selectable_value(slot, Some(m.clone()), name(m));
                        }
                    });
                    if ui.button(tr("Browse...")).clicked()
                        && let Some(f) = rfd::FileDialog::new().add_filter(tr("GGML model"), &["bin"]).pick_file()
                    {
                        *slot = Some(f);
                    }
                    ui.end_row();
                }
            });
            ui.label(tr("Download a model:"));
            for &(stem, desc, _) in MODELS {
                self.download_row(ui, stem, desc);
            }
            ui.checkbox(&mut self.cfg.use_gpu, tr("Use GPU"))
                .on_hover_text(tr("Vulkan: Intel UHD/Iris Xe/Arc, AMD and NVIDIA. CUDA: NVIDIA only. Fallback to CPU"));
            if !cfg!(any(feature = "vulkan", feature = "cuda", feature = "metal")) {
                ui.small(tr("This build has no GPU support"));
            }

            ui.separator();
            ui.heading(tr("Omnilingual ASR (Meta, local)"));
            let omni: Vec<PathBuf> = std::fs::read_dir(config::models_dir().join("omniasr"))
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "pt"))
                .collect();
            egui::Grid::new("omni_models").num_columns(2).show(ui, |ui| {
                for (mode, label) in [(Mode::Fast, tr("Fast mode")), (Mode::Accurate, tr("Accurate mode"))] {
                    ui.label(label);
                    let selected = self.cfg.omni_model(mode).as_deref().map_or(tr("None").into(), name);
                    let slot = self.cfg.omni_model_mut(mode);
                    egui::ComboBox::from_id_salt(("omni", label)).selected_text(selected).width(260.0).show_ui(ui, |ui| {
                        for m in &omni {
                            ui.selectable_value(slot, Some(m.clone()), name(m));
                        }
                    });
                    ui.end_row();
                }
            });
            for &(stem, desc, _) in config::OMNI_MODELS {
                self.download_row(ui, stem, desc);
            }
            ui.checkbox(&mut self.cfg.refine_timing, tr("Refine word timing of the other engines"))
                .on_hover_text(tr("Re-times Whisper and API words with the Omnilingual model (fast model preferred): more precise seeking and speaker labels. Whisper often starts words after a pause too early"));
            ui.small(tr("Uses the GPU when \"Use GPU\" is on. The language is detected automatically"));

            ui.separator();
            ui.heading(tr("Speaker identification"));
            let spk: Vec<PathBuf> = std::fs::read_dir(config::models_dir().join("speaker"))
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "pt"))
                .collect();
            ui.horizontal(|ui| {
                ui.label(tr("Model"));
                let selected = config::speaker_model(&self.cfg.speaker_model).as_deref().map_or(tr("None").into(), name);
                egui::ComboBox::from_id_salt("speaker_model").selected_text(selected).width(260.0).show_ui(ui, |ui| {
                    for m in &spk {
                        ui.selectable_value(&mut self.cfg.speaker_model, Some(m.clone()), name(m));
                    }
                });
            });
            for &(stem, desc) in config::SPEAKER_MODELS {
                self.download_row(ui, stem, desc);
            }
            ui.small(tr("Used with \"Speakers\" in the toolbar"));

            ui.separator();
            ui.heading(tr("OpenAI-compatible API"));
            egui::Grid::new("openai").num_columns(2).show(ui, |ui| {
                ui.label(tr("Base URL"));
                ui.text_edit_singleline(&mut self.cfg.openai_url);
                ui.end_row();
                ui.label(tr("API key"));
                ui.add(egui::TextEdit::singleline(&mut self.cfg.openai_key).password(true).hint_text(tr("or OPENAI_API_KEY")));
                ui.end_row();
                ui.label(tr("Model"));
                ui.text_edit_singleline(&mut self.cfg.openai_model);
                ui.end_row();
            });
            ui.small(tr("Needs word timestamps (e.g. OpenAI whisper-1, Groq whisper-large-v3)."));

            ui.separator();
            ui.heading(tr("Google Cloud Speech-to-Text"));
            egui::Grid::new("google").num_columns(2).show(ui, |ui| {
                ui.label(tr("Base URL"));
                ui.text_edit_singleline(&mut self.cfg.google_url);
                ui.end_row();
                ui.label(tr("API key"));
                ui.add(egui::TextEdit::singleline(&mut self.cfg.google_key).password(true).hint_text(tr("or GOOGLE_API_KEY")));
                ui.end_row();
            });
            ui.small(tr("Requires choosing a language (e.g. en, de, ja)."));

            ui.separator();
            ui.checkbox(&mut self.cfg.transcribe_denoised, tr("Transcribe the voice-isolated audio"))
                .on_hover_text(tr("Can help with very noisy recordings, the original usually works best"));
            ui.horizontal(|ui| {
                ui.label(tr("Interface language"));
                let current = if self.cfg.ui_language == "auto" { tr("System") } else { i18n::LOCALES.iter().find(|l| l.code == self.cfg.ui_language).map_or("?", |l| l.name) };
                egui::ComboBox::from_id_salt("ui_language").selected_text(current).show_ui(ui, |ui| {
                    let mut changed = ui.selectable_value(&mut self.cfg.ui_language, "auto".into(), tr("System")).changed();
                    for l in i18n::LOCALES {
                        changed |= ui.selectable_value(&mut self.cfg.ui_language, l.code.into(), l.name).changed();
                    }
                    if changed {
                        i18n::set(&self.cfg.ui_language);
                    }
                });
            });
            ui.horizontal(|ui| {
                ui.label(tr("Text size"));
                ui.add(egui::Slider::new(&mut self.cfg.font_size, 10.0..=32.0).fixed_decimals(0));
            });
            ui.small(tr("Keys: Esc play/pause - F1/F2 back/forward 3 s - F3/F4 slower/faster - Ctrl+S save"));
            ui.small(tr("API keys are stored in plain text in the config file"));
        });
        if self.show_settings && !open {
            self.cfg.save();
        }
        self.show_settings = open;
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll();
        self.shortcuts(&ctx);

        let dropped = ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_owned()));
        if let Some(p) = dropped {
            self.open(p, &ctx);
        }

        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            self.toolbar(ui);
            ui.add_space(2.0);
            self.waveforms(ui);
            self.transport(ui);
            ui.add_space(2.0);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(d) = &self.download {
                    let (done, total) = (d.done.load(Ordering::Relaxed), d.total.load(Ordering::Relaxed).max(1));
                    ui.label(trf("Downloading {0}...", &[d.stem]));
                    ui.add(egui::ProgressBar::new(done as f32 / total as f32).desired_width(160.0).show_percentage());
                }
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                    if let Some(p) = &self.project {
                        ui.weak(trf("Saved to {0}", &[&p.display().to_string()]));
                    }
                });
            });
        });
        egui::CentralPanel::default().show(ui, |ui| self.editor(ui));
        if self.show_settings {
            self.settings(&ctx);
        }
        if self.model_prompt {
            self.model_prompt(&ctx);
        }

        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let screen = ctx.content_rect();
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("drop")));
            painter.rect_filled(screen, 0.0, Color32::from_black_alpha(160));
            painter.text(screen.center(), egui::Align2::CENTER_CENTER, tr("Drop to open"), egui::FontId::proportional(28.0), Color32::WHITE);
        }

        let pos = self.pos();
        if self.player().is_some_and(|p| p.playing()) {
            if self.cfg.follow && (pos < self.view.start || pos > self.view.start + self.view.len * 0.95) {
                self.view.start = pos - self.view.len * 0.05;
                self.view.clamp(self.duration());
            }
            ctx.request_repaint();
        }
        if self.download.is_some() {
            ctx.request_repaint_after(Duration::from_millis(200));
        }
        if self.dirty.is_some() {
            ctx.request_repaint_after(Duration::from_secs(1));
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.cancel_job();
        self.save();
        self.cfg.save();
    }
}
