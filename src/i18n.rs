//! UI translations, using English text as the key and falling back to English.
//! New languages need to be added to 'LOCALES' below.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Locale {
    /// BCP 47 language code
    pub code: &'static str,
    /// Name of the language in that language
    pub name: &'static str,
    data: &'static str,
}

pub const LOCALES: &[Locale] = &[
    Locale { code: "en", name: "English", data: "{}" },
    // Locale { code: "de", name: "Deutsch", data: include_str!("../locales/de.json") },
];

static ACTIVE: AtomicUsize = AtomicUsize::new(0);

fn tables() -> &'static [HashMap<&'static str, &'static str>] {
    static TABLES: OnceLock<Vec<HashMap<&'static str, &'static str>>> = OnceLock::new();
    TABLES.get_or_init(|| {
        LOCALES
            .iter()
            .map(|l| {
                let map: HashMap<String, String> = serde_json::from_str(l.data).unwrap_or_else(|e| panic!("locales/{}.json: {e}", l.code));
                // leaked once per language so lookups return &'static str
                map.into_iter().filter(|(_, v)| !v.is_empty()).map(|(k, v)| (&*k.leak(), &*v.leak())).collect()
            })
            .collect()
    })
}

/// Translates 'text' into the active language
pub fn tr(text: &'static str) -> &'static str {
    tables()[ACTIVE.load(Ordering::Relaxed)].get(text).copied().unwrap_or(text)
}

/// Translates and fills '{0}', '{1}' placeholders
pub fn trf(text: &'static str, args: &[&str]) -> String {
    let mut s = tr(text).to_string();
    for (i, a) in args.iter().enumerate() {
        s = s.replace(&format!("{{{i}}}"), a);
    }
    s
}

/// Marks a constant for translation without translating it yet
pub const fn n_(text: &'static str) -> &'static str {
    text
}

/// Activates the best match for 'code', "auto" uses the system language
pub fn set(code: &str) {
    let code = if code == "auto" { system_language() } else { code.to_string() };
    let find = |c: &str| LOCALES.iter().position(|l| l.code.eq_ignore_ascii_case(c));
    let i = find(&code).or_else(|| find(code.split(['-', '_']).next().unwrap_or(""))).unwrap_or(0);
    ACTIVE.store(i, Ordering::Relaxed);
}

/// System UI language as a BCP 47 code
fn system_language() -> String {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Globalization::GetUserDefaultLocaleName;
        let mut buf = [0u16; 85];
        // SAFETY: the buffer length is passed and the call writes at most that many u16s
        let n = unsafe { GetUserDefaultLocaleName(buf.as_mut_ptr(), buf.len() as i32) };
        if n > 1 {
            return String::from_utf16_lossy(&buf[..n as usize - 1]);
        }
    }
    ["LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .flat_map(|v| v.split(':').map(str::to_owned).collect::<Vec<_>>())
        .map(|v| v.split(['.', '@']).next().unwrap_or("").replace('_', "-"))
        .find(|v| !v.is_empty() && v != "C" && v != "POSIX")
        .unwrap_or_else(|| "en".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// String literals passed to tr/trf/n_ in the sources
    fn source_keys() -> BTreeSet<String> {
        let mut keys = BTreeSet::new();
        for e in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src")).unwrap().flatten() {
            let src = std::fs::read_to_string(e.path()).unwrap();
            // skip this module's own docs and tests
            let src = if e.file_name() == "i18n.rs" { "" } else { src.as_str() };
            for marker in ["tr(\"", "trf(\"", "n_(\""] {
                for (at, _) in src.match_indices(marker) {
                    if src[..at].ends_with(|c: char| c.is_alphanumeric() || c == '_') {
                        continue;
                    }
                    keys.insert(parse_literal(&src[at + marker.len()..]));
                }
            }
        }
        keys
    }

    /// Reads a Rust string literal body up to the closing quote
    fn parse_literal(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            match c {
                '"' => return out,
                '\\' => match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some(c) => out.push(c),
                    None => break,
                },
                c => out.push(c),
            }
        }
        panic!("unterminated string literal");
    }

    fn placeholders(s: &str) -> BTreeSet<String> {
        (0..10).map(|i| format!("{{{i}}}")).filter(|p| s.contains(p.as_str())).collect()
    }

    #[test]
    fn template_lists_all_strings() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/locales/template.json");
        let keys = source_keys();
        assert!(keys.len() > 50, "found only {} strings", keys.len());
        let template: BTreeSet<String> = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&s).ok())
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        if template != keys {
            if std::env::var_os("UPDATE_LOCALES").is_some() {
                let map: serde_json::Map<_, _> = keys.into_iter().map(|k| (k, serde_json::Value::String(String::new()))).collect();
                std::fs::write(path, serde_json::to_string_pretty(&map).unwrap() + "\n").unwrap();
            } else {
                let missing: Vec<_> = keys.difference(&template).collect();
                let stale: Vec<_> = template.difference(&keys).collect();
                panic!("locales/template.json is outdated (run UPDATE_LOCALES=1 cargo test)\nmissing: {missing:?}\nstale: {stale:?}");
            }
        }
    }

    #[test]
    fn locales_match_template() {
        let keys = source_keys();
        for (l, table) in LOCALES.iter().zip(tables()) {
            for (k, v) in table {
                assert!(keys.contains(*k), "{}: unknown key {k:?}", l.code);
                assert_eq!(placeholders(k), placeholders(v), "{}: placeholders differ in {k:?}", l.code);
            }
        }
    }

    #[test]
    fn lookup_and_fallback() {
        set("en");
        assert_eq!(tr("Play"), "Play");
        assert_eq!(trf("Continue after {0}", &["1:00"]), "Continue after 1:00");
        set("xx-YY");
        assert_eq!(ACTIVE.load(Ordering::Relaxed), 0);
    }
}
