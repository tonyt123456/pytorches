//! Plugin discovery, loading and the global plugin registry.

use libloading::Library;
use pytorches_plugin_abi::*;
use std::ffi::{CStr, c_char};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

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

pub fn plugins() -> Vec<Arc<Plugin>> {
    REGISTRY.read().unwrap().clone()
}

pub fn find_plugin(name: &str) -> Option<Arc<Plugin>> {
    REGISTRY.read().unwrap().iter().find(|p| p.name == name).cloned()
}

fn register_vtable(vt: *const PluginVTable, lib: Option<Library>) -> Result<Arc<Plugin>, String> {
    if vt.is_null() {
        return Err("plugin entry returned null".into());
    }
    // SAFETY: abi_version is the first field of every ABI version.
    let version = unsafe { (*vt).abi_version };
    if version != ABI_VERSION {
        return Err(format!("plugin ABI version {version}, core expects {ABI_VERSION}"));
    }
    let vt: &'static PluginVTable = unsafe { &*vt };
    let name = unsafe { CStr::from_ptr(vt.name as *const c_char) }.to_string_lossy().into_owned();

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

/// Loads every `pytorches_plugin_*` shared library in `dir`.
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
            let r = load_plugin_file(&p).map(|pl| pl.name.clone());
            (p, r)
        })
        .collect()
}

/// Loads plugins from `$PYTORCHES_PLUGIN_DIR` (if set), else `./plugins/bin`, else
/// `<exe dir>/plugins`. Safe to call repeatedly.
pub fn discover() -> Vec<(PathBuf, Result<String, String>)> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("PYTORCHES_PLUGIN_DIR") {
        dirs.extend(std::env::split_paths(&d));
    } else {
        dirs.push(PathBuf::from("plugins/bin"));
        if let Some(exe_dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
            dirs.push(exe_dir.join("plugins"));
        }
    }
    dirs.iter().flat_map(|d| load_plugin_dir(d)).collect()
}
