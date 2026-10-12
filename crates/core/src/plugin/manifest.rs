//! Gate 1: the optional plugin manifest, checked without loading any code.

use pytorches_plugin_abi::ABI_VERSION;
use std::path::{Path, PathBuf};

/// Optional `pytorches_plugin_<name>.json` next to a plugin library. Everything is optional;
/// unknown keys are ignored. Example:
/// `{"name": "cuda", "vendor": "nvidia", "abi": 4, "os": ["windows", "linux"], "arch": ["x86_64"]}`
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Manifest {
    pub name: Option<String>,
    pub vendor: Option<String>,
    pub abi: Option<u32>,
    /// Accepted values of `std::env::consts::OS` ("windows", "linux", "macos"); empty = any.
    pub os: Vec<String>,
    /// Accepted values of `std::env::consts::ARCH` ("x86_64", "aarch64"); empty = any.
    pub arch: Vec<String>,
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Manifest, String> {
        let Json::Obj(fields) = Json::parse(text)? else {
            return Err("manifest must be a JSON object".into());
        };
        let mut m = Manifest::default();
        for (key, val) in fields {
            match (key.as_str(), val) {
                ("name", Json::Str(v)) => m.name = Some(v),
                ("vendor", Json::Str(v)) => m.vendor = Some(v),
                ("abi", Json::Num(v)) if v >= 0.0 && v.fract() == 0.0 => m.abi = Some(v as u32),
                ("os", v) => m.os = v.strings("os")?,
                ("arch", v) => m.arch = v.strings("arch")?,
                _ => {}
            }
        }
        Ok(m)
    }

    /// Whether this plugin applies to the running core and machine. Loads nothing.
    pub fn check(&self) -> Result<(), String> {
        if let Some(abi) = self.abi
            && abi != ABI_VERSION
        {
            return Err(format!("plugin targets ABI {abi}, core expects {ABI_VERSION}"));
        }
        if !self.os.is_empty() && !self.os.iter().any(|o| o == std::env::consts::OS) {
            return Err(format!("plugin is for {}, this is {}", self.os.join("/"), std::env::consts::OS));
        }
        if !self.arch.is_empty() && !self.arch.iter().any(|a| a == std::env::consts::ARCH) {
            return Err(format!("plugin is for {}, this is {}", self.arch.join("/"), std::env::consts::ARCH));
        }
        Ok(())
    }
}

/// Where the manifest for `lib` lives: same directory, `pytorches_plugin_<name>.json`.
pub fn manifest_path(lib: &Path) -> PathBuf {
    let stem = lib.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    lib.with_file_name(format!("{}.json", stem.strip_prefix("lib").unwrap_or(stem)))
}

/// Checks the manifest beside `lib`, if there is one. No manifest means nothing to check.
pub(super) fn check_manifest(lib: &Path) -> Result<(), String> {
    let Ok(text) = std::fs::read_to_string(manifest_path(lib)) else { return Ok(()) };
    Manifest::parse(&text).and_then(|m| m.check()).map_err(|e| format!("{}: {e}", lib.display()))
}

/// Just enough JSON for manifests (no serde dependency in the core).
#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn parse(text: &str) -> Result<Json, String> {
        let mut p = JsonParser { s: text.as_bytes(), i: 0 };
        let v = p.value()?;
        p.ws();
        if p.i != p.s.len() {
            return Err("trailing characters after JSON value".into());
        }
        Ok(v)
    }

    fn strings(self, key: &str) -> Result<Vec<String>, String> {
        let bad = || format!("'{key}' must be a string or list of strings");
        match self {
            Json::Str(s) => Ok(vec![s]),
            Json::Arr(items) => items
                .into_iter()
                .map(|v| match v {
                    Json::Str(s) => Ok(s),
                    _ => Err(bad()),
                })
                .collect(),
            _ => Err(bad()),
        }
    }
}

struct JsonParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("invalid JSON at byte {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek() {
            Some(b'{') => {
                self.i += 1;
                let mut fields = Vec::new();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(fields));
                }
                loop {
                    self.ws();
                    let key = self.string()?;
                    self.eat(b':')?;
                    fields.push((key, self.value()?));
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(fields));
                        }
                        _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(c) if c.is_ascii_digit() || c == b'-' => {
                let start = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[start..self.i])
                    .ok()
                    .and_then(|t| t.parse().ok())
                    .map(Json::Num)
                    .ok_or_else(|| format!("invalid number at byte {start}"))
            }
            _ => Err(format!("invalid JSON at byte {}", self.i)),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.s.get(self.i) != Some(&b'"') {
            return Err(format!("expected string at byte {}", self.i));
        }
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.s.get(self.i) else { return Err("unterminated string".into()) };
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|_| "invalid UTF-8 in string".into()),
                b'\\' => {
                    let Some(&e) = self.s.get(self.i) else { return Err("unterminated escape".into()) };
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let hex = self.s.get(self.i..self.i + 4).and_then(|h| std::str::from_utf8(h).ok());
                            // Surrogate halves are not worth decoding for manifest text.
                            let ch = hex.and_then(|h| u32::from_str_radix(h, 16).ok()).and_then(char::from_u32);
                            self.i += 4;
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.unwrap_or('\u{fffd}').encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err(format!("bad escape at byte {}", self.i)),
                    }
                }
                _ => out.push(c),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_ignores_unknown_keys() {
        let m = Manifest::parse(
            r#"{"name": "cuda", "vendor": "nvidia", "abi": 4, "os": ["windows", "linux"],
                "arch": "x86_64", "nested": {"a": [1, 2.5e3, null, true]}, "note": "café \"x\""}"#,
        )
        .unwrap();
        assert_eq!(m.name.as_deref(), Some("cuda"));
        assert_eq!(m.vendor.as_deref(), Some("nvidia"));
        assert_eq!(m.abi, Some(4));
        assert_eq!(m.os, ["windows", "linux"]);
        assert_eq!(m.arch, ["x86_64"]);
    }

    #[test]
    fn rejects_bad_json() {
        for bad in ["", "[]", "{", r#"{"a": }"#, r#"{"a": 1} x"#, r#"{"os": 5}"#, r#"{"a": "unterminated}"#] {
            assert!(Manifest::parse(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn check_filters_without_loading() {
        assert!(Manifest::default().check().is_ok());
        let wrong_abi = Manifest { abi: Some(ABI_VERSION + 1), ..Default::default() };
        assert!(wrong_abi.check().unwrap_err().contains("ABI"));
        let wrong_os = Manifest { os: vec!["plan9".into()], ..Default::default() };
        assert!(wrong_os.check().unwrap_err().contains("plan9"));
        let wrong_arch = Manifest { arch: vec!["m68k".into()], ..Default::default() };
        assert!(wrong_arch.check().is_err());
        let ok = Manifest {
            abi: Some(ABI_VERSION),
            os: vec![std::env::consts::OS.into()],
            arch: vec![std::env::consts::ARCH.into()],
            ..Default::default()
        };
        assert!(ok.check().is_ok());
    }

    #[test]
    fn path_strips_lib_prefix_and_extension() {
        assert_eq!(manifest_path(Path::new("d/pytorches_plugin_cuda.dll")), Path::new("d/pytorches_plugin_cuda.json"));
        assert_eq!(manifest_path(Path::new("d/libpytorches_plugin_cuda.so")), Path::new("d/pytorches_plugin_cuda.json"));
    }

    #[test]
    fn missing_manifest_is_fine_and_bad_manifest_skips_the_plugin() {
        let dir = std::env::temp_dir().join(format!("pytorches-manifest-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let lib = dir.join("pytorches_plugin_x.dll");
        assert!(check_manifest(&lib).is_ok());
        std::fs::write(manifest_path(&lib), r#"{"os": ["plan9"]}"#).unwrap();
        assert!(check_manifest(&lib).unwrap_err().contains("plan9"));
        std::fs::write(manifest_path(&lib), "not json").unwrap();
        assert!(check_manifest(&lib).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
