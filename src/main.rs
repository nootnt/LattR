#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod align;
mod app;
mod audio;
mod config;
mod ggml;
mod i18n;
mod omni;
mod player;
mod speaker;
mod transcribe;
mod transcript;
mod waveform;

use std::sync::Arc;

/// System fonts covering scripts egui's built-in fonts lack (CJK, Indic, Thai, …).
fn fallback_fonts() -> Vec<(std::path::PathBuf, u32)> {
    let mut out: Vec<(std::path::PathBuf, u32)> = Vec::new();
    if cfg!(windows) {
        let dir = std::path::PathBuf::from(std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into())).join("Fonts");
        for f in ["msyh.ttc", "YuGothM.ttc", "malgun.ttf", "Nirmala.ttc", "Nirmala.ttf", "leelawui.ttf", "segoeui.ttf", "sylfaen.ttf", "arial.ttf", "seguisym.ttf"] {
            out.push((dir.join(f), 0));
        }
    } else {
        // fc-list only lists fonts that really cover the language (fc-match always returns something)
        for lang in ["ja", "ko", "zh-cn", "hi", "bn", "ta", "th", "ar", "he", "ka", "hy", "am", "my", "km"] {
            let Ok(o) = std::process::Command::new("fc-list").args(["-f", "%{file}|%{index}|%{style}\n", &format!(":lang={lang}")]).output() else { break };
            let list = String::from_utf8_lossy(&o.stdout);
            // styles list the primary name first, e.g. "Bold,Regular"
            let score = |style: &str, file: &str| (style.split(',').next() == Some("Regular")) as u8 * 4 + file.contains("Sans") as u8 * 2 + file.contains("Noto") as u8;
            let best = list
                .lines()
                .filter_map(|l| {
                    let mut p = l.splitn(3, '|');
                    Some((p.next()?, p.next()?.parse().unwrap_or(0), p.next().unwrap_or("")))
                })
                .max_by_key(|&(file, _, style)| score(style, file));
            if let Some((file, idx, _)) = best {
                out.push((file.into(), idx));
            }
        }
    }
    out.dedup();
    out.into_iter().filter(|(p, _)| p.exists()).fold(Vec::new(), |mut v, f| {
        if !v.contains(&f) {
            v.push(f);
        }
        v
    })
}

fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    for (i, (path, index)) in fallback_fonts().into_iter().enumerate() {
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let name = format!("fallback{i}");
        let mut data = egui::FontData::from_owned(bytes);
        data.index = index;
        fonts.font_data.insert(name.clone(), Arc::new(data));
        for fam in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            fonts.families.entry(fam).or_default().push(name.clone());
        }
    }
    ctx.set_fonts(fonts);
}

fn main() -> eframe::Result {
    audio::clean_cache();
    i18n::set(&config::Config::load().ui_language);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 760.0])
            .with_min_inner_size([520.0, 400.0])
            .with_drag_and_drop(true)
            .with_title("LattR")
            .with_app_id("lattr")
            .with_icon(egui::IconData { rgba: include_bytes!("../assets/icon64.rgba").to_vec(), width: 64, height: 64 }),
        ..Default::default()
    };
    eframe::run_native(
        "LattR",
        options,
        Box::new(|cc| {
            setup_fonts(&cc.egui_ctx);
            let mut app = app::App::new();
            if let Some(path) = std::env::args_os().nth(1) {
                app.open(path.into(), &cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    )
}
