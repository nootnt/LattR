//! Editable transcript text with per-word timestamps that follow edits

use crate::i18n::trf;
use egui::TextBuffer;
use egui::text::CharIndex;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// A timed word, 'start..end' in chars of 'Transcript::text', times in seconds
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub t0: f64,
    pub t1: f64,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Transcript {
    pub text: String,
    pub spans: Vec<Span>,
    /// Display names by speaker number
    #[serde(default)]
    pub speakers: Vec<String>,
    #[serde(default)]
    last_speaker: Option<u32>,
    #[serde(skip)]
    chars: Option<usize>,
}

/// Pause length that starts a new paragraph
const PARAGRAPH_GAP: f64 = 2.5;

fn byte_at(s: &str, c: usize) -> usize {
    s.char_indices().nth(c).map_or(s.len(), |(b, _)| b)
}

impl Transcript {
    fn char_len(&mut self) -> usize {
        *self.chars.get_or_insert_with(|| self.text.chars().count())
    }

    /// Appends a word, starting a new labelled paragraph on speaker change
    pub fn push_word(&mut self, word: &str, t0: f64, t1: f64, speaker: Option<u32>) {
        let trimmed = word.trim();
        if trimmed.is_empty() {
            return;
        }
        let len = self.char_len();
        let sep = if let Some(sp) = speaker.filter(|_| speaker != self.last_speaker) {
            self.last_speaker = speaker;
            while self.speakers.len() <= sp as usize {
                let n = self.speakers.len() + 1;
                self.speakers.push(trf("Speaker {0}", &[&n.to_string()]));
            }
            let brk = if self.text.is_empty() || self.text.ends_with("\n\n") { "" } else if self.text.ends_with('\n') { "\n" } else { "\n\n" };
            format!("{brk}{}: ", self.speakers[sp as usize])
        } else {
            match self.spans.last() {
                _ if self.text.is_empty() || self.text.ends_with('\n') => "",
                Some(last) if t0 - last.t1 >= PARAGRAPH_GAP => "\n\n",
                _ if word.starts_with(char::is_whitespace) && !self.text.ends_with(' ') => " ",
                _ => "",
            }
            .to_string()
        };
        let sep = sep.as_str();
        self.text.push_str(sep);
        self.text.push_str(trimmed);
        let start = len + sep.chars().count();
        let end = start + trimmed.chars().count();
        self.chars = Some(end);
        self.spans.push(Span { start, end, t0, t1: t1.max(t0) });
    }

    /// Makes a loaded transcript consistent, so a damaged file cannot crash the editor
    pub fn sanitize(&mut self) {
        let n = self.text.chars().count();
        self.chars = Some(n);
        self.spans.retain(|s| s.start <= s.end && s.end <= n && s.t0.is_finite() && s.t1.is_finite());
        self.spans.sort_by_key(|s| s.start);
        let mut last_end = 0;
        self.spans.retain(|s| {
            let ok = s.start >= last_end;
            if ok {
                last_end = s.end;
            }
            ok
        });
        if self.last_speaker.is_some_and(|l| l as usize >= self.speakers.len()) {
            self.last_speaker = None;
        }
    }

    /// Labels words from 'from' on, starting a paragraph at each speaker change
    pub fn apply_speakers(&mut self, from: f64, speaker_of: impl Fn(f64, f64) -> Option<u32>) {
        let first = self.spans.partition_point(|s| s.t0 < from);
        let labels: Vec<Option<u32>> = self.spans.iter().enumerate().map(|(i, s)| if i >= first && s.end > s.start { speaker_of(s.t0, s.t1) } else { None }).collect();
        let mut changes: Vec<(usize, u32)> = Vec::new();
        // continue from the labels of earlier words, if any
        let mut cur = if first > 0 { self.last_speaker } else { None };
        for (s, l) in self.spans.iter().zip(&labels) {
            if let Some(l) = l
                && cur != Some(*l)
            {
                changes.push((s.start, *l));
                cur = Some(*l);
            }
        }
        for &(_, sp) in &changes {
            while self.speakers.len() <= sp as usize {
                let n = self.speakers.len() + 1;
                self.speakers.push(trf("Speaker {0}", &[&n.to_string()]));
            }
        }
        // back to front so earlier char positions stay valid
        for &(at, sp) in changes.iter().rev() {
            let before: String = self.text.chars().take(at).collect();
            let trailing = before.chars().rev().take_while(|c| *c == ' ').count();
            let kept = &before[..before.len() - trailing];
            let brk = if kept.is_empty() || kept.ends_with("\n\n") { "" } else if kept.ends_with('\n') { "\n" } else { "\n\n" };
            if trailing > 0 {
                self.delete_char_range(CharIndex(at - trailing)..CharIndex(at));
            }
            let label = format!("{brk}{}: ", self.speakers[sp as usize]);
            self.insert_text(&label, CharIndex(at - trailing));
        }
        if let Some(l) = labels.iter().rev().flatten().next() {
            self.last_speaker = Some(*l);
        }
    }

    /// Renames a speaker and every "Old: " label at the start of a line
    pub fn rename_speaker(&mut self, speaker: usize, name: &str) {
        let Some(old) = self.speakers.get(speaker).cloned() else { return };
        let name = name.trim();
        if name.is_empty() || name == old {
            return;
        }
        let label = format!("{old}: ");
        let starts: Vec<usize> = self
            .text
            .match_indices(&label)
            .map(|(b, _)| b)
            .filter(|&b| b == 0 || self.text[..b].ends_with('\n'))
            .map(|b| self.text[..b].chars().count())
            .collect();
        let old_len = old.chars().count();
        // back to front so earlier positions stay valid
        for c in starts.into_iter().rev() {
            self.delete_char_range(CharIndex(c)..CharIndex(c + old_len));
            self.insert_text(name, CharIndex(c));
        }
        self.speakers[speaker] = name.to_string();
    }

    /// Word at or before char 'c'
    pub fn span_at_char(&self, c: usize) -> Option<&Span> {
        let i = self.spans.partition_point(|s| s.start <= c);
        self.spans.get(i.checked_sub(1)?)
    }

    /// Index of the word being spoken at time 't'
    pub fn span_at_time(&self, t: f64) -> Option<usize> {
        let i = self.spans.partition_point(|s| s.t0 <= t).checked_sub(1)?;
        let s = &self.spans[i];
        (t <= s.t1 + 0.5 && s.end > s.start).then_some(i)
    }

    pub fn slice(&self, r: Range<usize>) -> &str {
        let a = byte_at(&self.text, r.start);
        let b = a + byte_at(&self.text[a..], r.end - r.start);
        &self.text[a..b]
    }

    /// Text of spans in 'range', in one pass over the string
    pub fn words(&self, range: Range<usize>) -> Vec<(Span, &str)> {
        let spans = &self.spans[range];
        let Some(first) = spans.first() else { return Vec::new() };
        let base = byte_at(&self.text, first.start);
        let mut iter = self.text[base..].char_indices().map(|(b, _)| base + b).chain(std::iter::once(self.text.len()));
        let mut at = first.start;
        let mut next_byte = |c: usize, at: &mut usize| {
            let b = iter.nth(c - *at).unwrap_or(self.text.len());
            *at = c + 1;
            b
        };
        let mut out = Vec::with_capacity(spans.len());
        let mut last = (usize::MAX, 0);
        for s in spans {
            // spans are sorted by start, ends never pass the next start
            let a = if s.start == last.0 { last.1 } else { next_byte(s.start, &mut at) };
            let b = if s.end == s.start { a } else { next_byte(s.end, &mut at) };
            last = (s.end, b);
            out.push((*s, &self.text[a..b]));
        }
        out
    }

    pub fn to_srt(&self) -> String {
        let mut out = String::new();
        let mut cue: Vec<Span> = Vec::new();
        let mut n = 1;
        let mut flush = |cue: &mut Vec<Span>, out: &mut String| {
            if let (Some(a), Some(b)) = (cue.first(), cue.last()) {
                let text = self.slice(a.start..b.end).split_whitespace().collect::<Vec<_>>().join(" ");
                out.push_str(&format!("{n}\n{} --> {}\n{text}\n\n", srt_time(a.t0), srt_time(b.t1)));
                n += 1;
            }
            cue.clear();
        };
        for s in self.spans.iter().filter(|s| s.end > s.start) {
            if let (Some(a), Some(b)) = (cue.first(), cue.last()) {
                let chars = s.end - a.start;
                let sentence_end = self.slice(b.start..b.end).ends_with(['.', '?', '!', '。', '？', '！']);
                if s.t1 - a.t0 > 6.0 || chars > 84 || s.t0 - b.t1 > 1.0 || (sentence_end && chars > 30) {
                    flush(&mut cue, &mut out);
                }
            }
            cue.push(*s);
        }
        flush(&mut cue, &mut out);
        out
    }

    /// Maps old char positions after deleting 'a..b'
    fn delete_spans(&mut self, a: usize, b: usize) {
        let map = |x: usize| if x <= a { x } else if x >= b { x - (b - a) } else { a };
        let mut out: Vec<Span> = Vec::with_capacity(self.spans.len());
        for s in &self.spans {
            let n = Span { start: map(s.start), end: map(s.end), ..*s };
            // collapse words deleted entirely into one empty span so retyping keeps a time
            if let Some(last) = out.last_mut()
                && n.start == n.end
                && last.start == n.start
                && last.end == n.end
            {
                last.t1 = n.t1;
                continue;
            }
            out.push(n);
        }
        self.spans = out;
    }

    fn insert_spans(&mut self, at: usize, n: usize, text: &str) {
        // the inserted text joins: an emptied word, the word it is typed into, or the word it extends
        let joins = !text.starts_with(char::is_whitespace);
        let find = |f: &dyn Fn(&Span) -> bool| self.spans.iter().position(f);
        let target = find(&|s| s.start == at && s.end == at)
            .or_else(|| find(&|s| s.start < at && at < s.end))
            .or_else(|| find(&|s| joins && s.start < at && s.end == at));
        for (i, s) in self.spans.iter_mut().enumerate() {
            if Some(i) == target {
                s.end += n;
            } else if s.start >= at {
                s.start += n;
                s.end += n;
            }
        }
    }

}

impl TextBuffer for Transcript {
    fn is_mutable(&self) -> bool {
        true
    }

    fn as_str(&self) -> &str {
        &self.text
    }

    fn insert_text(&mut self, text: &str, char_index: CharIndex) -> usize {
        let at = char_index.0;
        let n = text.chars().count();
        let b = byte_at(&self.text, at);
        self.text.insert_str(b, text);
        self.chars = self.chars.map(|c| c + n);
        self.insert_spans(at, n, text);
        n
    }

    fn delete_char_range(&mut self, r: Range<CharIndex>) {
        let (a, b) = (r.start.0, r.end.0);
        if a >= b {
            return;
        }
        let ba = byte_at(&self.text, a);
        let bb = ba + byte_at(&self.text[ba..], b - a);
        self.text.drain(ba..bb);
        self.chars = None;
        self.delete_spans(a, b);
    }

    /// Applies undo/redo as a char diff, keeping the timing of untouched words
    fn replace_with(&mut self, text: &str) {
        let old: Vec<char> = self.text.chars().collect();
        let new: Vec<char> = text.chars().collect();
        let p = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
        let max_s = old.len().min(new.len()) - p;
        let s = old.iter().rev().zip(new.iter().rev()).take(max_s).take_while(|(a, b)| a == b).count();
        let (a, b) = (&old[p..old.len() - s], &new[p..new.len() - s]);
        // right to left so earlier positions stay valid
        for (ra, rb) in diff(a, b).into_iter().rev() {
            self.delete_char_range(CharIndex(p + ra.start)..CharIndex(p + ra.end));
            let ins: String = b[rb.clone()].iter().collect();
            if !ins.is_empty() {
                self.insert_text(&ins, CharIndex(p + ra.start));
            }
        }
    }

    fn type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self>()
    }
}

/// Hunks (old range, new range) turning 'a' into 'b' via LCS, with a single hunk for large inputs
fn diff(a: &[char], b: &[char]) -> Vec<(Range<usize>, Range<usize>)> {
    if a.is_empty() && b.is_empty() {
        return Vec::new();
    }
    if a.is_empty() || b.is_empty() || a.len() * b.len() > 4_000_000 {
        return vec![(0..a.len(), 0..b.len())];
    }
    let w = b.len() + 1;
    let mut lcs = vec![0u32; (a.len() + 1) * w];
    for i in (0..a.len()).rev() {
        for j in (0..b.len()).rev() {
            lcs[i * w + j] = if a[i] == b[j] { lcs[(i + 1) * w + j + 1] + 1 } else { lcs[(i + 1) * w + j].max(lcs[i * w + j + 1]) };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut hunks: Vec<(Range<usize>, Range<usize>)> = Vec::new();
    while i < a.len() || j < b.len() {
        if i < a.len() && j < b.len() && a[i] == b[j] {
            i += 1;
            j += 1;
            continue;
        }
        let (si, sj) = (i, j);
        while (i < a.len() || j < b.len()) && !(i < a.len() && j < b.len() && a[i] == b[j]) {
            if j >= b.len() || (i < a.len() && lcs[(i + 1) * w + j] >= lcs[i * w + j + 1]) {
                i += 1;
            } else {
                j += 1;
            }
        }
        hunks.push((si..i, sj..j));
    }
    hunks
}

pub fn srt_time(t: f64) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> Transcript {
        let mut t = Transcript::default();
        t.push_word(" Hello", 1.0, 1.5, None);
        t.push_word(" world.", 1.6, 2.0, None);
        t.push_word(" Again", 10.0, 10.5, None);
        t
    }

    fn words(t: &Transcript) -> Vec<&str> {
        t.spans.iter().map(|s| t.slice(s.start..s.end)).collect()
    }

    #[test]
    fn push_and_paragraphs() {
        let t = t();
        assert_eq!(t.text, "Hello world.\n\nAgain");
        assert_eq!(words(&t), ["Hello", "world.", "Again"]);
        assert_eq!(t.span_at_char(7).unwrap().t0, 1.6);
        assert_eq!(t.span_at_char(5).unwrap().t0, 1.0);
        assert_eq!(t.span_at_time(1.7), Some(1));
        assert_eq!(t.span_at_time(5.0), None);
    }

    #[test]
    fn typing_inside_and_after_word() {
        let mut t = t();
        t.insert_text("p", CharIndex(5)); // "Hellop"
        t.insert_text("X", CharIndex(1)); // "HXellop"
        assert_eq!(words(&t), ["HXellop", "world.", "Again"]);
        t.insert_text(" new", CharIndex(7));
        assert_eq!(t.text, "HXellop new world.\n\nAgain");
        assert_eq!(words(&t), ["HXellop", "world.", "Again"]);
    }

    #[test]
    fn replacing_word_keeps_time() {
        let mut t = t();
        // select "world." and type over it, like TextEdit does
        t.delete_char_range(CharIndex(6)..CharIndex(12));
        t.insert_text("earth!", CharIndex(6));
        assert_eq!(words(&t), ["Hello", "earth!", "Again"]);
        assert_eq!(t.spans[1].t0, 1.6);
    }

    #[test]
    fn deleting_across_words_merges() {
        let mut t = t();
        t.delete_char_range(CharIndex(0)..CharIndex(12));
        t.insert_text("Hi", CharIndex(0));
        assert_eq!(t.text, "Hi\n\nAgain");
        assert_eq!(t.spans[0].t0, 1.0);
        assert_eq!(t.spans[0].t1, 2.0);
        assert_eq!(words(&t), ["Hi", "Again"]);
    }

    #[test]
    fn undo_replace_with() {
        let mut t = t();
        let orig = t.text.clone();
        t.insert_text("abc", CharIndex(5));
        t.replace_with(&orig);
        assert_eq!(t.text, orig);
        assert_eq!(words(&t), ["Hello", "world.", "Again"]);
    }

    #[test]
    fn german_editing_and_undo() {
        let mut t = Transcript::default();
        for (i, w) in [" Grüß", " Gott,", " Herr", " Müller."].iter().enumerate() {
            t.push_word(w, i as f64, i as f64 + 0.5, None);
        }
        let before = t.text.clone();
        // "Müller" -> "Müllers" typed at the end, "Grüß" -> "Grüße"
        t.insert_text("s", CharIndex(22));
        t.insert_text("e", CharIndex(4));
        assert_eq!(t.text, "Grüße Gott, Herr Müllers.");
        assert_eq!(words(&t), ["Grüße", "Gott,", "Herr", "Müllers."]);
        t.replace_with(&before);
        assert_eq!(t.text, before);
        assert_eq!(words(&t), ["Grüß", "Gott,", "Herr", "Müller."]);
        assert_eq!(t.spans.iter().map(|s| s.t0).collect::<Vec<_>>(), [0.0, 1.0, 2.0, 3.0]);
    }

    fn german() -> Transcript {
        let mut t = Transcript::default();
        t.push_word(" Grüße,", 0.5, 1.2, None);
        t.push_word(" Welt!", 1.3, 2.0, None);
        t.push_word(" Wie", 6.0, 6.4, None);
        t.push_word(" geht's?", 6.5, 6.9, None);
        t
    }

    #[test]
    fn german_layout_and_lookup() {
        let t = german();
        assert_eq!(t.text, "Grüße, Welt!\n\nWie geht's?");
        assert_eq!(words(&t), ["Grüße,", "Welt!", "Wie", "geht's?"]);
        assert_eq!(t.span_at_char(7).unwrap().t0, 1.3);
        assert_eq!(t.span_at_char(6).unwrap().t0, 0.5);
        assert_eq!(t.span_at_time(6.45), Some(2));
    }

    #[test]
    fn srt_german() {
        let srt = german().to_srt();
        assert_eq!(srt, "1\n00:00:00,500 --> 00:00:02,000\nGrüße, Welt!\n\n2\n00:00:06,000 --> 00:00:06,900\nWie geht's?\n\n");
    }

    #[test]
    fn speaker_turns_and_rename() {
        let mut t = Transcript::default();
        t.push_word(" Hello", 0.0, 0.5, Some(0));
        t.push_word(" there.", 0.5, 1.0, Some(0));
        t.push_word(" Hi!", 1.2, 1.5, Some(1));
        t.push_word(" Bye.", 2.0, 2.4, Some(0));
        assert_eq!(t.text, "Speaker 1: Hello there.\n\nSpeaker 2: Hi!\n\nSpeaker 1: Bye.");
        assert_eq!(words(&t), ["Hello", "there.", "Hi!", "Bye."]);
        t.rename_speaker(0, "Nino");
        assert_eq!(t.text, "Nino: Hello there.\n\nSpeaker 2: Hi!\n\nNino: Bye.");
        // timing stays attached to the words
        assert_eq!(words(&t), ["Hello", "there.", "Hi!", "Bye."]);
        assert_eq!(t.span_at_char(t.text.chars().count() - 1).unwrap().t0, 2.0);
        assert_eq!(t.speakers, ["Nino", "Speaker 2"]);
    }

    #[test]
    fn speakers_applied_afterwards() {
        let mut t = Transcript::default();
        for (w, t0) in [(" Hello", 0.0), (" there.", 0.5), (" Hi!", 1.2), (" Yes.", 5.0)] {
            t.push_word(w, t0, t0 + 0.4, None);
        }
        assert_eq!(t.text, "Hello there. Hi!\n\nYes.");
        t.apply_speakers(0.0, |t0, _| Some(if (1.0..3.0).contains(&t0) { 1 } else { 0 }));
        assert_eq!(t.text, "Speaker 1: Hello there.\n\nSpeaker 2: Hi!\n\nSpeaker 1: Yes.");
        assert_eq!(words(&t), ["Hello", "there.", "Hi!", "Yes."]);
        assert_eq!(t.spans[2].t0, 1.2);
        // words arriving later continue the labelling
        t.push_word(" Again.", 5.5, 5.9, Some(0));
        t.push_word(" No.", 6.5, 6.9, Some(1));
        assert_eq!(t.text, "Speaker 1: Hello there.\n\nSpeaker 2: Hi!\n\nSpeaker 1: Yes. Again.\n\nSpeaker 2: No.");
    }

    #[test]
    fn sanitize_bad_file() {
        let mut t: Transcript = serde_json::from_str(
            r#"{"text":"ab cd","spans":[{"start":3,"end":5,"t0":1.0,"t1":2.0},{"start":0,"end":9,"t0":0.0,"t1":1.0},{"start":4,"end":2,"t0":0.0,"t1":1.0},{"start":0,"end":2,"t0":0.0,"t1":0.5}],"last_speaker":7}"#,
        )
        .unwrap();
        t.sanitize();
        assert_eq!(words(&t), ["ab", "cd"]);
        assert!(t.last_speaker.is_none());
        let all = t.words(0..t.spans.len());
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn unicode() {
        let mut t = Transcript::default();
        t.push_word("こんにちは", 0.0, 0.5, None);
        t.push_word("世界", 0.5, 1.0, None);
        assert_eq!(t.text, "こんにちは世界");
        t.insert_text("、", CharIndex(5));
        assert_eq!(words(&t), ["こんにちは、", "世界"]);
    }
}
