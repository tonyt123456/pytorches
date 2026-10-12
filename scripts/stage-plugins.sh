#!/usr/bin/env bash
# Builds every plugin and copies the shared libraries into the Python package so maturin can
# bundle them in the wheel. Run from anywhere; used by CI and for local wheel builds.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/crates/python/python/pytorches/plugins"
rm -rf "$dest"; mkdir -p "$dest"
for dir in "$root"/plugins/*/; do
    name="$(basename "$dir")"
    [ -f "$dir/Cargo.toml" ] || continue
    [ "$name" = bin ] && continue
    cargo build --release -p "pytorches-plugin-$name" --manifest-path "$root/Cargo.toml"
    # The manifest travels beside the library so the loader can skip a plugin without loading it.
    [ -f "$dir/manifest.json" ] && cp "$dir/manifest.json" "$dest/pytorches_plugin_${name//-/_}.json"
done
shopt -s nullglob
# Copy only the plugins built above, so stale libraries left in target/release are not bundled.
for dir in "$root"/plugins/*/; do
    name="$(basename "$dir")"
    [ -f "$dir/Cargo.toml" ] || continue
    [ "$name" = bin ] && continue
    lib="pytorches_plugin_${name//-/_}"
    for f in "$root/target/release/$lib".{dll,so,dylib} "$root/target/release/lib$lib".{so,dylib}; do
        [ -f "$f" ] && cp "$f" "$dest/"
    done
done
ls -l "$dest"
