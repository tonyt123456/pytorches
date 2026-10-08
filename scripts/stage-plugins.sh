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
done
shopt -s nullglob
for f in "$root"/target/release/pytorches_plugin_*.{dll,so,dylib} "$root"/target/release/libpytorches_plugin_*.{so,dylib}; do
    cp "$f" "$dest/"
done
ls -l "$dest"
