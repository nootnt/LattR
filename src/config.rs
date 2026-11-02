use crate::i18n::n_;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const APP: &str = "lattr";
pub const GOOGLE_URL: &str = "https://speech.googleapis.com/v1";

/// A 'portable.txt' next to the executable keeps all data in 'data/' beside it
fn portable_dir() -> Option<PathBuf> {
    let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    dir.join("portable.txt").exists().then(|| dir.join("data"))
}

fn base_dir(kind: &str) -> PathBuf {
    if let Some(dir) = portable_dir() {
        let dir = if kind == "data" { dir } else { dir.join(kind) };
        private_dir(&dir);
        return dir;
    }
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let home = env("HOME").or_else(|| env("USERPROFILE")).unwrap_or_else(std::env::temp_dir);
    let dir = if cfg!(windows) {
        let appdata = if kind == "config" { env("APPDATA") } else { env("LOCALAPPDATA") };
        appdata.unwrap_or(home)
    } else if cfg!(target_os = "macos") {
        let sub = if kind == "cache" { "Library/Caches" } else { "Library/Application Support" };
        home.join(sub)
    } else {
        match kind {
            "config" => env("XDG_CONFIG_HOME").unwrap_or_else(|| home.join(".config")),
            "cache" => env("XDG_CACHE_HOME").unwrap_or_else(|| home.join(".cache")),
            _ => env("XDG_DATA_HOME").unwrap_or_else(|| home.join(".local/share")),
        }
    };
    let dir = dir.join(APP);
    let dir = if cfg!(windows) && kind == "cache" { dir.join("cache") } else { dir };
    private_dir(&dir);
    dir
}

/// Creates 'dir' readable only by the user
pub fn private_dir(dir: &Path) {
    let _ = std::fs::create_dir_all(dir);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
    }
}

pub fn cache_dir() -> PathBuf {
    base_dir("cache")
}

pub fn models_dir() -> PathBuf {
    let dir = base_dir("data").join("models");
    private_dir(&dir);
    dir
}

fn config_path() -> PathBuf {
    base_dir("config").join("config.json")
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendKind {
    Whisper,
    Omni,
    OpenAi,
    Google,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Fast,
    Accurate,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Config {
    pub backend: BackendKind,
    pub mode: Mode,
    #[serde(alias = "model")]
    pub model_fast: Option<PathBuf>,
    pub model_accurate: Option<PathBuf>,
    pub use_gpu: bool,
    pub language: String,
    pub transcribe_denoised: bool,
    pub openai_url: String,
    pub openai_key: String,
    pub openai_model: String,
    pub google_url: String,
    pub google_key: String,
    #[serde(alias = "omni_model")]
    pub omni_fast: Option<PathBuf>,
    pub omni_accurate: Option<PathBuf>,
    pub detect_speakers: bool,
    /// 0 = estimate
    pub num_speakers: usize,
    pub speaker_model: Option<PathBuf>,
    pub refine_timing: bool,
    pub speed: f32,
    pub volume: f32,
    pub follow: bool,
    pub font_size: f32,
    pub ui_language: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: BackendKind::Whisper,
            mode: Mode::Fast,
            model_fast: None,
            model_accurate: None,
            use_gpu: true,
            language: "auto".into(),
            transcribe_denoised: false,
            openai_url: "https://api.openai.com/v1".into(),
            openai_key: String::new(),
            openai_model: "whisper-1".into(),
            google_url: GOOGLE_URL.into(),
            google_key: String::new(),
            omni_fast: None,
            omni_accurate: None,
            detect_speakers: false,
            num_speakers: 0,
            speaker_model: None,
            refine_timing: true,
            speed: 1.0,
            volume: 1.0,
            follow: true,
            font_size: 16.0,
            ui_language: "auto".into(),
        }
    }
}

impl Config {
    pub fn load() -> Self {
        std::fs::read(config_path())
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    pub fn model(&self, mode: Mode) -> &Option<PathBuf> {
        match mode {
            Mode::Fast => &self.model_fast,
            Mode::Accurate => &self.model_accurate,
        }
    }

    pub fn omni_model_mut(&mut self, mode: Mode) -> &mut Option<PathBuf> {
        match mode {
            Mode::Fast => &mut self.omni_fast,
            Mode::Accurate => &mut self.omni_accurate,
        }
    }

    pub fn omni_model(&self, mode: Mode) -> Option<PathBuf> {
        let chosen = match mode {
            Mode::Fast => &self.omni_fast,
            Mode::Accurate => &self.omni_accurate,
        };
        chosen.clone().filter(|m| m.exists()).or_else(|| OMNI_MODELS.iter().filter(|m| m.2 == mode).map(|m| model_path(m.0)).find(|p| p.exists()))
    }

    pub fn model_mut(&mut self, mode: Mode) -> &mut Option<PathBuf> {
        match mode {
            Mode::Fast => &mut self.model_fast,
            Mode::Accurate => &mut self.model_accurate,
        }
    }

    pub fn save(&self) {
        let Ok(json) = serde_json::to_vec_pretty(self) else { return };
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        // owner-only, the file holds API keys
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let path = config_path();
        if let Ok(mut f) = opts.open(&path) {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = f.set_permissions(std::fs::Permissions::from_mode(0o600));
            }
            let _ = std::io::Write::write_all(&mut f, &json);
        }
    }
}

/// Downloadable whisper.cpp models
pub const MODELS: &[(&str, &str, Mode)] = &[
    ("large-v3-turbo-q5_0", n_("Large v3 Turbo q5 - ~550 MB - fast, good quality"), Mode::Fast),
    ("large-v3-q5_0", n_("Large v3 q5 - ~1.1 GB - more accurate"), Mode::Accurate),
    ("large-v3", n_("Large v3 f16 - ~3.1 GB - most accurate, needs ~4 GB RAM/VRAM"), Mode::Accurate),
    ("small-q5_1", n_("Small q5 - ~200 MB - for slow computers"), Mode::Fast),
];

/// Meta Omnilingual ASR CTC checkpoints
pub const OMNI_MODELS: &[(&str, &str, Mode)] = &[
    ("omniASR-CTC-300M-v2", n_("Omnilingual 300M - ~1.3 GB - 1600+ languages, fast"), Mode::Fast),
    ("omniASR-CTC-1B-v2", n_("Omnilingual 1B - ~3.9 GB - much more accurate, ~2.5× slower"), Mode::Accurate),
];

/// WeSpeaker speaker-embedding models
pub const SPEAKER_MODELS: &[(&str, &str)] = &[
    ("wespeaker-resnet34-LM", n_("WeSpeaker ResNet34 - ~50 MB - fast, accurate enough for most recordings")),
    ("wespeaker-resnet221-LM", n_("WeSpeaker ResNet221 - ~120 MB - more accurate, ~3× slower")),
    ("wespeaker-resnet293-LM", n_("WeSpeaker ResNet293 - 134 MB - even more accurate, ~4× slower")),
];

/// The chosen speaker model, else the first downloaded one
pub fn speaker_model(chosen: &Option<PathBuf>) -> Option<PathBuf> {
    chosen.clone().filter(|p| p.exists()).or_else(|| SPEAKER_MODELS.iter().map(|m| model_path(m.0)).find(|p| p.exists()))
}

pub fn model_path(stem: &str) -> PathBuf {
    if stem.starts_with("wespeaker") {
        models_dir().join("speaker").join(format!("{stem}.pt"))
    } else if stem.starts_with("omniASR") {
        models_dir().join("omniasr").join(format!("{stem}.pt"))
    } else {
        models_dir().join(format!("ggml-{stem}.bin"))
    }
}

pub struct ModelFileData {
    pub url: &'static str,
    pub path: &'static str,
    pub sha256: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/models.rs"));

/// (url, destination, SHA-256) for every file a model needs
pub fn model_files(stem: &str) -> Vec<(String, PathBuf, &'static str)> {
    let Some((_, files)) = MODEL_FILES.iter().find(|(s, _)| *s == stem) else { return Vec::new() };
    files.iter().map(|f| (f.url.to_string(), models_dir().join(f.path), f.sha256)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_has_downloads() {
        for (stem, _, _) in MODELS.iter().chain(OMNI_MODELS.iter()) {
            assert!(MODEL_FILES.iter().any(|(s, _)| s == stem), "no download for {stem}");
        }
        for (stem, _) in SPEAKER_MODELS {
            assert!(MODEL_FILES.iter().any(|(s, _)| s == stem), "no download for {stem}");
        }
    }

    #[test]
    fn downloads_are_well_formed() {
        for (stem, files) in MODEL_FILES {
            assert!(!files.is_empty(), "no files for {stem}");
            for file in *files {
                assert_eq!(file.sha256.len(), 64, "{}", file.path);
                assert!(file.sha256.chars().all(|c| c.is_ascii_hexdigit()), "{}", file.path);
                assert!(!file.path.is_empty(), "{stem}");
            }
        }
    }
}
