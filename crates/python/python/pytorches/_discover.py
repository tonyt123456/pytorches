"""Where to look for plugin libraries: the bundled directory, then installed plugin packages.

A plugin distributed on its own (for example `pytorches-cuda`) is an ordinary Python package that
holds the plugin library and its manifest, and advertises itself with an entry point:

    [project.entry-points."pytorches.plugins"]
    cuda = "pytorches_cuda"

The value names a module; its directory is searched for `pytorches_plugin_*` libraries. The module
is located, not imported, so discovery runs none of the package's Python code. The native
library then still passes the loader's manifest and probe gates before it is trusted.
"""

import importlib.metadata
import importlib.util
import os
import pathlib

ENTRY_POINT_GROUP = "pytorches.plugins"


def entry_point_plugins():
    """Installed plugin packages: a list of `(entry point name, [directories])`."""
    found = []
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        module = ep.value.split(":")[0].strip()
        try:
            spec = importlib.util.find_spec(module)
        except Exception:  # a broken package must not break `import pytorches`
            continue
        if spec is None:
            continue
        if spec.submodule_search_locations:
            dirs = [pathlib.Path(p) for p in spec.submodule_search_locations]
        elif spec.origin:
            dirs = [pathlib.Path(spec.origin).parent]
        else:
            dirs = []
        dirs = [d for d in dirs if d.is_dir()]
        if dirs:
            found.append((ep.name, dirs))
    return found


def bundled_dir(package_dir):
    """The directory holding the plugins shipped with this package (wheel layout or a dev checkout)."""
    package_dir = pathlib.Path(package_dir)
    candidates = [package_dir / "plugins"] + [p / "plugins" / "bin" for p in list(package_dir.parents)[:5]]
    return next((c for c in candidates if c.is_dir()), None)


def plugin_dirs(package_dir):
    """Every directory to search, in priority order, without duplicates: bundled first, then
    installed plugin packages (so a bundled plugin wins over a third-party one of the same name)."""
    dirs = []
    bundled = bundled_dir(package_dir)
    if bundled:
        dirs.append(bundled)
    for _, ep_dirs in entry_point_plugins():
        dirs.extend(ep_dirs)
    seen, unique = set(), []
    for d in dirs:
        key = os.path.normcase(os.path.abspath(d))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def locate_plugins(package_dir):
    """Point the runtime at the plugin directories unless the user already did.

    `PYTORCHES_PLUGIN_DIR` (a path list) set by the user is authoritative: it replaces discovery,
    which is what you want when developing a plugin against a specific build.
    """
    if os.environ.get("PYTORCHES_PLUGIN_DIR"):
        return
    dirs = plugin_dirs(package_dir)
    if dirs:
        os.environ["PYTORCHES_PLUGIN_DIR"] = os.pathsep.join(str(d) for d in dirs)
