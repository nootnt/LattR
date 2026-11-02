//! Forced alignment: re-times recognized words with a CTC model's per-frame probabilities

use crate::omni::Omni;
use crate::transcribe::Word;
use std::collections::HashMap;

/// CTC frame length in seconds
const FRAME: f64 = 0.02;
/// Search margin around a group's original times
const MARGIN: f64 = 1.0;

pub struct Aligner {
    model: Omni,
    chars: HashMap<char, usize>,
    space: Option<usize>,
    blank: usize,
}

impl Aligner {
    pub fn new(model: Omni) -> Self {
        let (pieces, special) = model.pieces();
        let mut chars = HashMap::new();
        let mut space = None;
        for (i, p) in pieces.iter().enumerate() {
            if special[i] {
                continue;
            }
            let mut cs = p.chars();
            match (cs.next(), cs.next()) {
                (Some('▁') | Some(' '), None) => space = Some(i),
                (Some(c), None) => {
                    chars.entry(c).or_insert(i);
                }
                _ => {}
            }
        }
        // fairseq2 CTC uses the padding symbol as blank, fall back to the first special piece
        let blank = pieces.iter().position(|p| p == "<pad>").or_else(|| special.iter().position(|&s| s)).unwrap_or(0);
        Self { model, chars, space, blank }
    }

    /// Re-times 'words' (seconds relative to wav, 16 kHz) in place
    pub fn align(&self, wav: &[f32], words: &mut [Word]) -> Result<(), String> {
        if words.is_empty() {
            return Ok(());
        }
        let v = self.model.pieces().0.len();
        let em = self.model.emissions(wav)?;
        let frames = em.len() / v;
        // align in groups split at pauses, each within a window around its original times
        let mut start = 0;
        for i in 1..=words.len() {
            if i < words.len() && (words[i].t0 - words[i - 1].t1 < 1.0 && i - start < 40) {
                continue;
            }
            let group = &mut words[start..i];
            let f0 = ((group[0].t0 - MARGIN).max(0.0) / FRAME) as usize;
            let f1 = (((group[group.len() - 1].t1 + MARGIN) / FRAME) as usize).min(frames);
            if f1 > f0 {
                self.align_group(&em[f0 * v..f1 * v], v, f0, group);
            }
            start = i;
        }
        Ok(())
    }

    fn align_group(&self, em: &[f32], v: usize, offset: usize, words: &mut [Word]) {
        // token sequence with the word each token belongs to, spaces between words
        let mut tokens: Vec<(usize, Option<usize>)> = Vec::new();
        for (wi, w) in words.iter().enumerate() {
            let ids: Vec<usize> = w.text.to_lowercase().chars().filter(|c| c.is_alphanumeric()).filter_map(|c| self.chars.get(&c).copied()).collect();
            if ids.is_empty() {
                continue;
            }
            if let (Some(sp), false) = (self.space, tokens.is_empty()) {
                tokens.push((sp, None));
            }
            tokens.extend(ids.into_iter().map(|id| (id, Some(wi))));
        }
        let frames = em.len() / v;
        let ids: Vec<usize> = tokens.iter().map(|t| t.0).collect();
        let Some(path) = ctc_align(em, v, frames, &ids, self.blank) else { return };
        // first and last frame of every word's tokens
        let mut span: Vec<Option<(usize, usize)>> = vec![None; words.len()];
        for (f, tok) in path.iter().enumerate() {
            if let Some(t) = tok
                && let Some(wi) = tokens[*t].1
            {
                let s = span[wi].get_or_insert((f, f));
                s.1 = f;
            }
        }
        for (w, s) in words.iter_mut().zip(&span) {
            if let Some((a, b)) = s {
                w.t0 = (offset + a) as f64 * FRAME;
                w.t1 = (offset + b + 1) as f64 * FRAME;
            }
        }
        // words without alignable characters (numbers in digits, symbols) sit between their neighbours
        for i in 0..words.len() {
            if span[i].is_none() {
                let prev = (0..i).rev().find(|&j| span[j].is_some()).map(|j| words[j].t1);
                let next = (i + 1..words.len()).find(|&j| span[j].is_some()).map(|j| words[j].t0);
                if let (Some(p), Some(n)) = (prev, next) {
                    words[i].t0 = p.min(n);
                    words[i].t1 = n.max(p);
                } else if let Some(t) = prev.or(next) {
                    words[i].t0 = t;
                    words[i].t1 = t;
                }
            }
        }
    }
}

/// Viterbi CTC alignment of 'tokens' to 'frames' rows of log-probabilities
/// Returns the emitted token index per frame, or None when it cannot fit
fn ctc_align(em: &[f32], v: usize, frames: usize, tokens: &[usize], blank: usize) -> Option<Vec<Option<usize>>> {
    let n = tokens.len();
    if n == 0 || frames < n {
        return None;
    }
    // states: blank, t0, blank, t1, …, blank
    let states = 2 * n + 1;
    let label = |s: usize| if s.is_multiple_of(2) { blank } else { tokens[s / 2] };
    let mut score = vec![f32::NEG_INFINITY; states];
    let mut back = vec![0u8; frames * states];
    score[0] = em[blank];
    score[1] = em[tokens[0]];
    for f in 1..frames {
        let row = &em[f * v..(f + 1) * v];
        let mut next = vec![f32::NEG_INFINITY; states];
        for s in 0..states {
            let mut best = (score[s], 0u8);
            if s >= 1 && score[s - 1] > best.0 {
                best = (score[s - 1], 1);
            }
            // skip the blank between two different tokens
            if s >= 2 && s % 2 == 1 && label(s) != label(s - 2) && score[s - 2] > best.0 {
                best = (score[s - 2], 2);
            }
            if best.0 > f32::NEG_INFINITY {
                next[s] = best.0 + row[label(s)];
                back[f * states + s] = best.1;
            }
        }
        score = next;
    }
    let mut s = if score[states - 1] >= score[states - 2] { states - 1 } else { states - 2 };
    if score[s] == f32::NEG_INFINITY {
        return None;
    }
    let mut path = vec![None; frames];
    for f in (0..frames).rev() {
        path[f] = (s % 2 == 1).then_some(s / 2);
        if f > 0 {
            s -= back[f * states + s] as usize;
        }
    }
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viterbi_follows_emissions() {
        // vocabulary: 0 blank, 1 'a', 2 'b', frames favour: blank a a blank b blank
        let favour = [0, 1, 1, 0, 2, 0];
        let v = 3;
        let em: Vec<f32> = favour.iter().flat_map(|&k| (0..v).map(move |i| if i == k { -0.1 } else { -5.0 })).collect();
        let path = ctc_align(&em, v, favour.len(), &[1, 2], 0).unwrap();
        assert_eq!(path, [None, Some(0), Some(0), None, Some(1), None]);
        // repeated tokens need a blank between them
        let favour = [1, 0, 1, 0];
        let em: Vec<f32> = favour.iter().flat_map(|&k| (0..v).map(move |i| if i == k { -0.1 } else { -5.0 })).collect();
        assert_eq!(ctc_align(&em, v, 4, &[1, 1], 0).unwrap(), [Some(0), None, Some(1), None]);
        assert!(ctc_align(&em, v, 1, &[1, 2], 0).is_none());
    }
}
