//! Plugin discovery, loading and the global plugin registry.
//!
//! Loading a candidate library goes through three gates, cheapest first, so a plugin that does not
//! apply to this machine (or that crashes its driver) never touches the host process:
//!
//! 1. **Manifest** (optional `pytorches_plugin_<name>.json` beside the library): ABI version, OS and
//!    architecture are checked without loading any code. See [`Manifest`].
//! 2. **Probe** (only when [`set_probe_command`] was called): the library is loaded in a short-lived
//!    child process with a timeout and asked how many devices it sees. A crash, hang or "no
//!    devices" skips the plugin. Outcomes are cached on disk (see [`probe_cache_file`]).
//! 3. **Load** in the host process, which re-checks the ABI header and device count.
//!
//! Environment: `PYTORCHES_PROBE=off` disables gate 2, `PYTORCHES_PROBE_CACHE=off` disables the
//! cache, `PYTORCHES_PROBE_TIMEOUT=<seconds>` (default 20) bounds the child, and
//! `PYTORCHES_CACHE_DIR` relocates the cache.

use libloading::Library;
use pytorches_plugin_abi::*;
use std::ffi::{CStr, c_char};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

mod manifest;
mod probe;

pub use manifest::{Manifest, manifest_path};
pub use probe::{
    PROBE_CHILD_ENV, ProbeDevice, ProbeInfo, inspect_library, is_probe_child, probe_cache_file, probe_main,
    set_probe_command,
};

/// A loaded plugin. Plugins are never unloaded, so the vtable reference is `'static`.
pub struct Plugin {
    pub name: String,
    pub(crate) vt: &'static PluginVTable,
    _lib: Option<Library>,
}

impl Plugin {
    pub fn device_count(&self) -> u32 {
        unsafe { (self.vt.device_count)() }
    }

    pub(crate) fn last_error(&self) -> String {
        unsafe {
            let p = (self.vt.last_error)();
            if p.is_null() {
                "unknown error".into()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }
}

static REGISTRY: RwLock<Vec<Arc<Plugin>>> = RwLock::new(Vec::new());

/// One entry per candidate library from the most recent [`discover`] call.
pub type DiscoveryReport = Vec<(PathBuf, Result<String, String>)>;
static LAST_DISCOVERY: Mutex<DiscoveryReport> = Mutex::new(Vec::new());

/// What the last [`discover`] call tried: each library and whether it loaded (plugin name) or why not.
pub fn discovery_report() -> DiscoveryReport {
    LAST_DISCOVERY.lock().unwrap().clone()
}

pub fn plugins() -> Vec<Arc<Plugin>> {
    REGISTRY.read().unwrap().clone()
}

pub fn find_plugin(name: &str) -> Option<Arc<Plugin>> {
    REGISTRY.read().unwrap().iter().find(|p| p.name == name).cloned()
}

/// Validates the leading `(abi_version, struct_size)` header of a plugin table.
fn check_vtable<'a>(vt: *const PluginVTable) -> Result<&'a PluginVTable, String> {
    if vt.is_null() {
        return Err("plugin entry returned null".into());
    }
    // SAFETY: every ABI major version starts with two u32 fields: abi_version, struct_size.
    let header = unsafe { std::slice::from_raw_parts(vt as *const u32, 2) };
    let (version, size) = (header[0], header[1]);
    if version != ABI_VERSION {
        return Err(format!("plugin ABI version {version}, core expects {ABI_VERSION}"));
    }
    if size < VTABLE_SIZE {
        return Err(format!("plugin vtable is {size} bytes, core needs at least {VTABLE_SIZE}"));
    }
    Ok(unsafe { &*vt })
}

fn vtable_name(vt: &PluginVTable) -> String {
    unsafe { CStr::from_ptr(vt.name as *const c_char) }.to_string_lossy().into_owned()
}

fn register_vtable(vt: *const PluginVTable, lib: Option<Library>) -> Result<Arc<Plugin>, String> {
    let vt: &'static PluginVTable = check_vtable(vt)?;
    let name = vtable_name(vt);

    let mut reg = REGISTRY.write().unwrap();
    if let Some(existing) = reg.iter().find(|p| p.name == name) {
        return Ok(existing.clone()); // already registered; keep the first
    }
    if unsafe { (vt.device_count)() } == 0 {
        return Err(format!("plugin '{name}' found no devices"));
    }
    let plugin = Arc::new(Plugin { name, vt, _lib: lib });
    reg.push(plugin.clone());
    Ok(plugin)
}

/// Registers a plugin linked into the current binary (used by tests and embedders).
pub fn register_static(entry: EntryFn) -> Result<Arc<Plugin>, String> {
    register_vtable(unsafe { entry() }, None)
}

/// Loads one plugin shared library.
pub fn load_plugin_file(path: &Path) -> Result<Arc<Plugin>, String> {
    unsafe {
        let lib = Library::new(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let entry: EntryFn = *lib
            .get::<EntryFn>(ENTRY_SYMBOL)
            .map_err(|e| format!("{}: missing entry symbol: {e}", path.display()))?;
        register_vtable(entry(), Some(lib)).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Loads every `pytorches_plugin_*` shared library in `dir`, passing each through the manifest and
/// probe gates (see the module docs).
/// Returns one result per candidate file: the plugin name, or why it failed.
pub fn load_plugin_dir(dir: &Path) -> Vec<(PathBuf, Result<String, String>)> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            (name.starts_with(FILE_PREFIX) || name.starts_with(&format!("lib{FILE_PREFIX}")))
                && matches!(ext, "dll" | "so" | "dylib")
        })
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let r = load_candidate(&p).map(|pl| pl.name.clone());
            (p, r)
        })
        .collect()
}

/// Loads plugins from `$PYTORCHES_PLUGIN_DIR` (if set), else `./plugins/bin`, else
/// `<exe dir>/plugins`. Safe to call repeatedly.
pub fn discover() -> DiscoveryReport {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("PYTORCHES_PLUGIN_DIR") {
        dirs.extend(std::env::split_paths(&d));
    } else {
        dirs.push(PathBuf::from("plugins/bin"));
        if let Some(exe_dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
            dirs.push(exe_dir.join("plugins"));
        }
    }
    let report: DiscoveryReport = dirs.iter().flat_map(|d| load_plugin_dir(d)).collect();
    *LAST_DISCOVERY.lock().unwrap() = report.clone();
    report
}

/// Manifest check, then isolated probe (if enabled), then the in-process load.
fn load_candidate(path: &Path) -> Result<Arc<Plugin>, String> {
    manifest::check_manifest(path)?;
    if let Some(command) = probe::probe_command() {
        probe::probe_verdict(&command, path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    load_plugin_file(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vtable_header_is_validated() {
        let cpu = pytorches_plugin_cpu::pytorches_plugin_entry();
        let mut copy: Vec<u8> = unsafe { std::slice::from_raw_parts(cpu as *const u8, size_of::<PluginVTable>()) }.to_vec();
        let ptr = |b: &Vec<u8>| b.as_ptr() as *const PluginVTable;
        assert!(check_vtable(cpu).is_ok());
        assert!(check_vtable(std::ptr::null()).is_err());

        copy[0..4].copy_from_slice(&(ABI_VERSION + 1).to_le_bytes());
        assert!(check_vtable(ptr(&copy)).err().unwrap().contains("ABI version"));
        copy[0..4].copy_from_slice(&ABI_VERSION.to_le_bytes());
        copy[4..8].copy_from_slice(&(VTABLE_SIZE - 8).to_le_bytes());
        assert!(check_vtable(ptr(&copy)).err().unwrap().contains("bytes"));
        // A larger table (a newer revision with appended fields) is accepted.
        copy[4..8].copy_from_slice(&(VTABLE_SIZE + 64).to_le_bytes());
        assert!(check_vtable(ptr(&copy)).is_ok());
    }
}
