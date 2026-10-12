//! Gate 2: probing a plugin library in a child process, with an on-disk cache of the outcome.
//!
//! Vendor drivers can hang or crash while initializing. Loading an unknown library straight into
//! the host process would take the host down with it, so the host first asks a short-lived child
//! (see [`set_probe_command`]) to load the library and report what it sees.

use super::{check_vtable, vtable_name};
use libloading::Library;
use pytorches_plugin_abi::*;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Set in the environment of probe children so embedders can skip their normal start-up work.
pub const PROBE_CHILD_ENV: &str = "PYTORCHES_PROBE_CHILD";
const OUTPUT_TAG: &str = "@probe";
const POSITIVE_TTL: Duration = Duration::from_secs(24 * 3600);
const NEGATIVE_TTL: Duration = Duration::from_secs(10 * 60);
const CACHE_MAX_ENTRIES: usize = 200;

/// True inside a probe child (see [`set_probe_command`]).
pub fn is_probe_child() -> bool {
    std::env::var_os(PROBE_CHILD_ENV).is_some()
}

static PROBE_COMMAND: Mutex<Option<Vec<String>>> = Mutex::new(None);

/// Enables isolated probing. `command` is a program plus arguments; the plugin library path is
/// appended as the last argument. The child must print [`probe_main`]'s result to stdout. A Python
/// host passes `[sys.executable, "-c", "...print(_native.probe_plugin(sys.argv[1]))"]`.
/// An empty command disables isolation again.
pub fn set_probe_command(command: Vec<String>) {
    *PROBE_COMMAND.lock().unwrap() = if command.is_empty() { None } else { Some(command) };
}

/// The configured command, unless `PYTORCHES_PROBE=off`.
pub(super) fn probe_command() -> Option<Vec<String>> {
    if env_is_off("PYTORCHES_PROBE") {
        return None;
    }
    PROBE_COMMAND.lock().unwrap().clone()
}

fn env_is_off(var: &str) -> bool {
    matches!(std::env::var(var).ok().as_deref(), Some("off" | "0" | "false" | "no"))
}

/// What a library reported about itself when probed.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeInfo {
    pub name: String,
    pub abi_version: u32,
    pub plugin_version: u32,
    pub devices: Vec<ProbeDevice>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeDevice {
    pub name: String,
    pub kind: u32,
    pub total_memory: u64,
}

/// Reads identity and device list from a validated plugin table. Registers nothing.
pub(super) fn inspect_vtable(vt: *const PluginVTable) -> Result<ProbeInfo, String> {
    let vt = check_vtable(vt)?;
    let mut devices = Vec::new();
    for i in 0..unsafe { (vt.device_count)() } {
        let mut info = std::mem::MaybeUninit::<DeviceInfo>::zeroed();
        let st = unsafe { (vt.device_info)(i, info.as_mut_ptr()) };
        let info = unsafe { info.assume_init() };
        devices.push(if st == STATUS_OK {
            let bytes: Vec<u8> = info.name.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
            ProbeDevice { name: String::from_utf8_lossy(&bytes).into_owned(), kind: info.kind, total_memory: info.total_memory }
        } else {
            ProbeDevice { name: "unknown".into(), kind: KIND_OTHER, total_memory: MEMORY_UNKNOWN }
        });
    }
    Ok(ProbeInfo { name: vtable_name(vt), abi_version: vt.abi_version, plugin_version: vt.plugin_version, devices })
}

/// Loads `path` and queries it without registering it. This is what a probe child runs; it can
/// crash or hang if the vendor driver does, which is why it runs in a child.
pub fn inspect_library(path: &Path) -> Result<ProbeInfo, String> {
    unsafe {
        let lib = Library::new(path).map_err(|e| e.to_string())?;
        let entry: EntryFn = *lib.get::<EntryFn>(ENTRY_SYMBOL).map_err(|e| format!("missing entry symbol: {e}"))?;
        let info = inspect_vtable(entry());
        // Vendor runtimes are not always safe to unload, and the child exits right after.
        std::mem::forget(lib);
        info
    }
}

/// The text a probe child must print for `path`.
pub fn probe_main(path: &Path) -> String {
    format_probe(&inspect_library(path))
}

fn ascii_line(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_graphic() || c == ' ' { c } else { '?' }).collect()
}

fn format_probe(r: &Result<ProbeInfo, String>) -> String {
    match r {
        Err(e) => format!("{OUTPUT_TAG}\terror\t{}\n", ascii_line(e)),
        Ok(i) => {
            let mut out = format!("{OUTPUT_TAG}\tok\t{}\t{}\t{}\n", ascii_line(&i.name), i.abi_version, i.plugin_version);
            for d in &i.devices {
                out += &format!("{OUTPUT_TAG}\tdevice\t{}\t{}\t{}\n", d.kind, d.total_memory, ascii_line(&d.name));
            }
            out
        }
    }
}

/// Parses child output. Lines without the tag (driver chatter) are ignored.
fn parse_probe(text: &str) -> Result<ProbeInfo, String> {
    let mut info: Option<ProbeInfo> = None;
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(OUTPUT_TAG).and_then(|r| r.strip_prefix('\t')) else { continue };
        let mut f = rest.splitn(5, '\t');
        match f.next() {
            Some("error") => return Err(f.next().unwrap_or("probe failed").to_string()),
            Some("ok") => {
                let name = f.next().unwrap_or("").to_string();
                let abi_version = f.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                let plugin_version = f.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                info = Some(ProbeInfo { name, abi_version, plugin_version, devices: Vec::new() });
            }
            Some("device") => {
                let kind = f.next().and_then(|v| v.parse().ok()).unwrap_or(KIND_OTHER);
                let total_memory = f.next().and_then(|v| v.parse().ok()).unwrap_or(MEMORY_UNKNOWN);
                let name = f.next().unwrap_or("").to_string();
                if let Some(i) = info.as_mut() {
                    i.devices.push(ProbeDevice { name, kind, total_memory });
                }
            }
            _ => {}
        }
    }
    info.ok_or_else(|| NO_RESULT.to_string())
}

const NO_RESULT: &str = "probe produced no result";

fn probe_timeout() -> Duration {
    let secs = std::env::var("PYTORCHES_PROBE_TIMEOUT").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(20.0);
    Duration::from_secs_f64(secs.clamp(0.1, 600.0))
}

/// Runs `command <lib>` and returns what it reports. Kills the child after `timeout`; a nonzero
/// exit (driver crash) is an error.
fn probe_in_subprocess(command: &[String], lib: &Path, timeout: Duration) -> Result<ProbeInfo, String> {
    let mut cmd = Command::new(&command[0]);
    cmd.args(&command[1..])
        .arg(lib)
        .env(PROBE_CHILD_ENV, "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| format!("could not start probe '{}': {e}", command[0]))?;
    let mut stdout = child.stdout.take().expect("stdout is piped");
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stdout.read_to_string(&mut text);
        text
    });
    let start = Instant::now();
    let status = loop {
        match child.try_wait().map_err(|e| format!("probe wait failed: {e}"))? {
            Some(s) => break s,
            None if start.elapsed() > timeout => {
                // Not joining the reader: a grandchild could still hold the pipe open.
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("probe timed out after {:.1}s (driver hang?)", timeout.as_secs_f64()));
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    };
    let text = reader.join().unwrap_or_default();
    if !status.success() {
        return Err(match parse_probe(&text).err().filter(|e| e != NO_RESULT) {
            Some(detail) => format!("probe failed: {detail}"),
            None => format!("probe crashed ({status}); the vendor driver likely failed to initialize"),
        });
    }
    parse_probe(&text)
}

// ---- cache ------------------------------------------------------------------------

/// `$PYTORCHES_CACHE_DIR/probe-cache.tsv`, else the per-user cache directory, if one can be found.
pub fn probe_cache_file() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let dir = var("PYTORCHES_CACHE_DIR")
        .or_else(|| var("LOCALAPPDATA").map(|d| d.join("pytorches")))
        .or_else(|| var("XDG_CACHE_HOME").map(|d| d.join("pytorches")))
        .or_else(|| var("HOME").map(|d| d.join(".cache").join("pytorches")))?;
    Some(dir.join("probe-cache.tsv"))
}

/// Outcome stored per library: `Ok((plugin name, device count))` or the reason it was skipped.
type Verdict = Result<(String, usize), String>;

struct CacheEntry {
    key: String,
    stamp: u64,
    verdict: Verdict,
}

fn ttl(v: &Verdict) -> Duration {
    if v.is_ok() { POSITIVE_TTL } else { NEGATIVE_TTL }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Identifies one build of one library: a changed file is re-probed.
fn cache_key(lib: &Path) -> Option<String> {
    let meta = std::fs::metadata(lib).ok()?;
    let mtime = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_secs();
    let path = std::fs::canonicalize(lib).unwrap_or_else(|_| lib.to_path_buf());
    Some(format!("{}|{}|{}|{}", path.display(), meta.len(), mtime, ABI_VERSION))
}

fn read_cache(file: &Path) -> Vec<CacheEntry> {
    let Ok(text) = std::fs::read_to_string(file) else { return Vec::new() };
    text.lines()
        .filter_map(|line| {
            let mut f = line.splitn(5, '\t');
            let key = f.next()?.to_string();
            let stamp = f.next()?.parse().ok()?;
            let verdict = match f.next()? {
                "ok" => Ok((f.next()?.to_string(), f.next()?.parse().ok()?)),
                "err" => Err(f.next()?.to_string()),
                _ => return None,
            };
            Some(CacheEntry { key, stamp, verdict })
        })
        .collect()
}

fn write_cache(file: &Path, entries: &[CacheEntry]) {
    let text: String = entries
        .iter()
        .map(|e| match &e.verdict {
            Ok((name, n)) => format!("{}\t{}\tok\t{}\t{}\n", e.key, e.stamp, name.replace(['\t', '\n', '\r'], " "), n),
            Err(msg) => format!("{}\t{}\terr\t{}\n", e.key, e.stamp, msg.replace(['\t', '\n', '\r'], " ")),
        })
        .collect();
    if let Some(dir) = file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // Write-then-rename so a concurrent reader never sees a torn file. Losing a race loses an
    // entry, not correctness.
    let tmp = file.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, text).is_ok() && std::fs::rename(&tmp, file).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn cache_lookup(file: &Path, key: &str, now: u64) -> Option<Verdict> {
    let entry = read_cache(file).into_iter().rev().find(|e| e.key == key)?;
    (now.saturating_sub(entry.stamp) < ttl(&entry.verdict).as_secs()).then_some(entry.verdict)
}

fn cache_store(file: &Path, key: String, verdict: Verdict, now: u64) {
    let mut entries: Vec<CacheEntry> = read_cache(file)
        .into_iter()
        .filter(|e| e.key != key && now.saturating_sub(e.stamp) < ttl(&e.verdict).as_secs())
        .collect();
    entries.push(CacheEntry { key, stamp: now, verdict });
    let skip = entries.len().saturating_sub(CACHE_MAX_ENTRIES);
    write_cache(file, &entries[skip..]);
}

/// Probes `lib` in a child (or answers from the cache). `Ok` means: loads cleanly and sees devices.
pub(super) fn probe_verdict(command: &[String], lib: &Path) -> Result<(), String> {
    let cache_file = if env_is_off("PYTORCHES_PROBE_CACHE") { None } else { probe_cache_file() };
    verdict_with(cache_file.as_deref(), command, lib, probe_timeout()).map(|_| ())
}

fn verdict_with(cache_file: Option<&Path>, command: &[String], lib: &Path, timeout: Duration) -> Verdict {
    let cache = cache_file.zip(cache_key(lib));
    let now = now_secs();
    if let Some((file, key)) = &cache
        && let Some(v) = cache_lookup(file, key, now)
    {
        return v;
    }
    let verdict = probe_in_subprocess(command, lib, timeout).and_then(|info| {
        if info.devices.is_empty() {
            Err(format!("plugin '{}' found no devices", info.name))
        } else {
            Ok((info.name, info.devices.len()))
        }
    });
    if let Some((file, key)) = cache {
        cache_store(file, key, verdict.clone(), now);
    }
    verdict
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pytorches-probe-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fake_info(devices: usize) -> ProbeInfo {
        ProbeInfo {
            name: "fake".into(),
            abi_version: ABI_VERSION,
            plugin_version: 1,
            devices: (0..devices)
                .map(|i| ProbeDevice { name: format!("d{i}"), kind: KIND_OTHER, total_memory: 7 + i as u64 })
                .collect(),
        }
    }

    /// A command that prints `output` (tabs and newlines intact) and exits 0.
    fn echo(output: &str) -> Vec<String> {
        let lines: Vec<String> = ["noise before"].into_iter().map(String::from).chain(output.lines().map(String::from)).collect();
        if cfg!(windows) {
            let script = lines.iter().map(|l| format!("Write-Output '{l}'")).collect::<Vec<_>>().join("; ") + " #"; // '#': the appended library path becomes a comment
            vec!["powershell".into(), "-NoProfile".into(), "-Command".into(), script]
        } else {
            let mut cmd = vec!["sh".into(), "-c".into(), "printf '%s\\n' \"$@\"".into(), "sh".into()];
            cmd.extend(lines);
            cmd
        }
    }

    fn shell(unix: &str, windows: &str) -> Vec<String> {
        if cfg!(windows) {
            vec!["powershell".into(), "-NoProfile".into(), "-Command".into(), format!("{windows} #")]
        } else {
            vec!["sh".into(), "-c".into(), unix.into()]
        }
    }

    #[test]
    fn inspect_reports_cpu_identity_and_devices() {
        let info = inspect_vtable(pytorches_plugin_cpu::pytorches_plugin_entry()).unwrap();
        assert_eq!(info.name, "cpu");
        assert_eq!(info.abi_version, ABI_VERSION);
        assert!(!info.devices.is_empty());
        assert_eq!(info.devices[0].kind, KIND_CPU);
    }

    #[test]
    fn probe_output_round_trips_and_skips_chatter() {
        let mut info = fake_info(2);
        info.devices[0].name = "NVIDIA RTX\twith tab".into();
        info.devices[1].name = "caf\u{e9}".into();
        let text = format!("driver says hello\n{}trailing noise\n", format_probe(&Ok(info)));
        let back = parse_probe(&text).unwrap();
        assert_eq!(back.name, "fake");
        assert_eq!(back.devices.len(), 2);
        assert_eq!(back.devices[0].total_memory, 7);
        assert_eq!(back.devices[0].name, "NVIDIA RTX?with tab"); // sanitized to one ASCII line
        assert_eq!(back.devices[1].name, "caf?");

        assert_eq!(parse_probe(&format_probe(&Err("boom".into()))).unwrap_err(), "boom");
        assert_eq!(parse_probe("nothing useful\n").unwrap_err(), NO_RESULT);
    }

    #[test]
    fn cache_honors_ttls_and_prunes() {
        let dir = scratch("cache");
        let file = dir.join("probe-cache.tsv");
        assert_eq!(cache_lookup(&file, "k", 100), None);

        cache_store(&file, "good".into(), Ok(("cuda".into(), 2)), 1000);
        cache_store(&file, "bad".into(), Err("no\tdevices\nfound".into()), 1000);
        assert_eq!(cache_lookup(&file, "good", 1000 + 3600), Some(Ok(("cuda".into(), 2))));
        assert_eq!(cache_lookup(&file, "bad", 1000 + 60), Some(Err("no devices found".into())));
        // Failures are retried sooner than successes expire.
        assert_eq!(cache_lookup(&file, "bad", 1000 + NEGATIVE_TTL.as_secs()), None);
        assert!(cache_lookup(&file, "good", 1000 + NEGATIVE_TTL.as_secs()).is_some());
        assert_eq!(cache_lookup(&file, "good", 1000 + POSITIVE_TTL.as_secs()), None);

        // A newer entry for the same key replaces the old one; expired entries are dropped on write.
        cache_store(&file, "bad".into(), Ok(("rocm".into(), 1)), 1000 + POSITIVE_TTL.as_secs() + 5);
        let entries = read_cache(&file);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "bad");

        for i in 0..CACHE_MAX_ENTRIES + 20 {
            cache_store(&file, format!("k{i}"), Ok(("x".into(), 1)), 5000);
        }
        assert_eq!(read_cache(&file).len(), CACHE_MAX_ENTRIES);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_key_changes_with_the_file() {
        let dir = scratch("key");
        let lib = dir.join("pytorches_plugin_x.dll");
        std::fs::write(&lib, b"one").unwrap();
        let a = cache_key(&lib).unwrap();
        std::fs::write(&lib, b"longer contents").unwrap();
        assert_ne!(a, cache_key(&lib).unwrap());
        assert!(cache_key(&dir.join("missing")).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn subprocess_probe_reads_child_output() {
        let cmd = echo(&format_probe(&Ok(fake_info(1))));
        let info = probe_in_subprocess(&cmd, Path::new("ignored"), Duration::from_secs(30)).unwrap();
        assert_eq!(info.name, "fake");
        assert_eq!(info.devices.len(), 1);
        assert_eq!(info.devices[0].total_memory, 7);
    }

    #[test]
    fn subprocess_probe_reports_crash_hang_and_missing_program() {
        let crash = shell("exit 3", "exit 3");
        let e = probe_in_subprocess(&crash, Path::new("x"), Duration::from_secs(30)).unwrap_err();
        assert!(e.contains("crashed"), "{e}");

        let hang = shell("sleep 30", "Start-Sleep 30");
        let start = Instant::now();
        let e = probe_in_subprocess(&hang, Path::new("x"), Duration::from_millis(500)).unwrap_err();
        assert!(e.contains("timed out"), "{e}");
        assert!(start.elapsed() < Duration::from_secs(15));

        let missing = vec!["pytorches-no-such-program".to_string()];
        assert!(probe_in_subprocess(&missing, Path::new("x"), Duration::from_secs(1)).unwrap_err().contains("could not start"));
    }

    #[test]
    fn verdict_is_cached_per_library_build() {
        let dir = scratch("verdict");
        let lib = dir.join("pytorches_plugin_fake.dll");
        std::fs::write(&lib, b"stub").unwrap();
        let cache = dir.join("probe-cache.tsv");
        let secs = Duration::from_secs(30);

        let good = echo(&format_probe(&Ok(fake_info(2))));
        assert_eq!(verdict_with(Some(&cache), &good, &lib, secs), Ok(("fake".into(), 2)));
        // The second call must come from the cache: the "probe" would crash if it ran.
        let crash = shell("exit 3", "exit 3");
        assert_eq!(verdict_with(Some(&cache), &crash, &lib, secs), Ok(("fake".into(), 2)));
        // A rebuilt library (different size) is probed again.
        std::fs::write(&lib, b"a different build").unwrap();
        assert!(verdict_with(Some(&cache), &crash, &lib, secs).unwrap_err().contains("crashed"));
        // Without a cache every call probes.
        assert!(verdict_with(None, &crash, &lib, secs).is_err());
        assert!(verdict_with(None, &good, &lib, secs).is_ok());

        // A plugin that sees no devices is a (cached) failure.
        std::fs::write(&lib, b"third build, no devices").unwrap();
        let none = echo(&format_probe(&Ok(fake_info(0))));
        assert_eq!(verdict_with(Some(&cache), &none, &lib, secs), Err("plugin 'fake' found no devices".into()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
