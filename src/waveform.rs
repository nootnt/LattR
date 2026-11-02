use crate::audio::Envelope;
use egui::{Color32, PointerButton, Rect, Sense, Stroke, Ui, pos2, vec2};

/// Visible time window of the zoomed view, in seconds
#[derive(Clone, Copy)]
pub struct View {
    pub start: f64,
    pub len: f64,
}

impl View {
    pub fn clamp(&mut self, duration: f64) {
        self.len = self.len.clamp(0.5, duration.max(0.5));
        self.start = self.start.clamp(0.0, (duration - self.len).max(0.0));
    }

    fn x(&self, rect: Rect, t: f64) -> f32 {
        rect.left() + ((t - self.start) / self.len) as f32 * rect.width()
    }

    fn t(&self, rect: Rect, x: f32) -> f64 {
        self.start + ((x - rect.left()) / rect.width()) as f64 * self.len
    }
}

fn draw_wave(ui: &Ui, rect: Rect, env: &Envelope, view: View, color: Color32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let mid = rect.center().y;
    let per_px = view.len * 100.0 / rect.width() as f64; // 100 buckets per second
    let w = rect.width() as usize;
    for px in 0..w {
        let a = (view.start * 100.0 + px as f64 * per_px).floor().max(0.0) as usize;
        let b = ((view.start * 100.0 + (px + 1) as f64 * per_px).ceil() as usize).max(a + 1);
        let h = env.max(a, b).sqrt() * rect.height() * 0.48;
        if h > 0.3 {
            let x = rect.left() + px as f32 + 0.5;
            painter.line_segment([pos2(x, mid - h), pos2(x, mid + h)], Stroke::new(1.0, color));
        }
    }
}

fn playhead(ui: &Ui, rect: Rect, view: View, pos: f64) {
    let x = view.x(rect, pos);
    if (rect.left()..=rect.right()).contains(&x) {
        ui.painter_at(rect).vline(x, rect.y_range(), Stroke::new(2.0, ui.visuals().warn_fg_color));
    }
}

/// Left press seeks right away and left drag scrubs, so a wobbly click still seeks
fn scrub(ui: &Ui, resp: &egui::Response, rect: Rect, view: View, duration: f64) -> Option<f64> {
    let pressed = resp.is_pointer_button_down_on() && ui.input(|i| i.pointer.primary_pressed());
    let dragged = resp.dragged_by(PointerButton::Primary) && resp.drag_delta().x != 0.0;
    if pressed || dragged {
        return Some(view.t(rect, resp.interact_pointer_pos()?.x).clamp(0.0, duration));
    }
    None
}

/// Whole-file waveform. Left click/drag seeks, middle drag moves the zoom window, right drag selects a new zoom window
pub fn overview(ui: &mut Ui, env: &Envelope, duration: f64, pos: f64, view: &mut View, color: Color32) -> Option<f64> {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 44.0), Sense::click_and_drag());
    let full = View { start: 0.0, len: duration.max(0.001) };
    draw_wave(ui, rect, env, full, color);
    let sel = Rect::from_x_y_ranges(full.x(rect, view.start)..=full.x(rect, view.start + view.len), rect.y_range());
    ui.painter_at(rect).rect_filled(sel, 0.0, ui.visuals().selection.bg_fill.gamma_multiply(0.35));
    playhead(ui, rect, full, pos);

    if resp.dragged_by(PointerButton::Middle) {
        view.start += resp.drag_delta().x as f64 / rect.width() as f64 * full.len;
        view.clamp(duration);
    }
    let drag_id = resp.id.with("drag_start");
    if resp.drag_started_by(PointerButton::Secondary) && let Some(p) = resp.interact_pointer_pos()
    {
        ui.data_mut(|d| d.insert_temp(drag_id, full.t(rect, p.x)));
    }
    if resp.dragged_by(PointerButton::Secondary) && let (Some(p), Some(a)) = (resp.interact_pointer_pos(), ui.data(|d| d.get_temp::<f64>(drag_id)))
    {
        let b = full.t(rect, p.x);
        *view = View { start: a.min(b), len: (a - b).abs().max(0.5) };
        view.clamp(duration);
    }
    scrub(ui, &resp, rect, full, duration)
}

/// Zoomed waveform with word labels. Left click/drag seeks, middle drag pans, scroll zooms
pub fn detail(ui: &mut Ui, env: &Envelope, duration: f64, pos: f64, view: &mut View, color: Color32, words: &[(f64, &str)]) -> Option<f64> {
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 110.0), Sense::click_and_drag());
    draw_wave(ui, rect, env, *view, color);
    let painter = ui.painter_at(rect);
    let text_color = ui.visuals().strong_text_color();

    // time ticks
    let step = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0, 3600.0]
        .into_iter()
        .find(|s| view.len / s <= 12.0)
        .unwrap_or(7200.0);
    let mut t = (view.start / step).ceil() * step;
    while t < view.start + view.len {
        let x = view.x(rect, t);
        painter.vline(x, rect.top()..=rect.top() + 6.0, Stroke::new(1.0, ui.visuals().weak_text_color()));
        painter.text(pos2(x + 2.0, rect.top()), egui::Align2::LEFT_TOP, crate::app::fmt_time(t, step < 1.0), egui::FontId::monospace(10.0), ui.visuals().weak_text_color());
        t += step;
    }

    if view.len <= 120.0 {
        let mut last_x = f32::MIN;
        for &(t0, w) in words {
            let x = view.x(rect, t0);
            if x < rect.left() - 100.0 || x > rect.right() || x < last_x {
                continue;
            }
            painter.vline(x, rect.bottom() - 18.0..=rect.bottom(), Stroke::new(1.0, text_color.gamma_multiply(0.5)));
            let r = painter.text(pos2(x + 2.0, rect.bottom() - 2.0), egui::Align2::LEFT_BOTTOM, w, egui::FontId::proportional(12.0), text_color);
            last_x = r.right() + 4.0;
        }
    }
    playhead(ui, rect, *view, pos);

    if resp.hovered() {
        let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
        let factor = zoom as f64 * (-(scroll.y as f64) / 200.0).exp();
        if factor != 1.0 && let Some(p) = resp.hover_pos()
        {
            let anchor = view.t(rect, p.x);
            view.len /= factor;
            view.start = anchor - (anchor - view.start) / factor;
        }
        if scroll.x != 0.0 {
            view.start -= scroll.x as f64 / rect.width() as f64 * view.len;
        }
        view.clamp(duration);
    }
    if resp.dragged_by(PointerButton::Middle) {
        view.start -= resp.drag_delta().x as f64 / rect.width() as f64 * view.len;
        view.clamp(duration);
    }
    scrub(ui, &resp, rect, *view, duration)
}

pub fn placeholder(ui: &mut Ui, h: f32, text: &str) {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(13.0), ui.visuals().weak_text_color());
}
