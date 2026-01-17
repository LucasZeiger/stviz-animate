#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist="$root/dist/linux/stviz-animate"
archive="$root/dist/stviz-animate-ubuntu.tar.gz"

cargo build --release

rm -rf "$dist"
mkdir -p "$dist"

cp "$root/target/release/stviz-animate" "$dist/"
cp -R "$root/python" "$dist/python"

if [[ -d "$root/ffmpeg" ]]; then
  cp -R "$root/ffmpeg" "$dist/ffmpeg"
fi
if [[ -f "$root/bin/ffmpeg" ]]; then
  mkdir -p "$dist/bin"
  cp "$root/bin/ffmpeg" "$dist/bin/ffmpeg"
fi

mkdir -p "$root/dist"
tar -czf "$archive" -C "$root/dist/linux" "stviz-animate"
echo "Ubuntu bundle created at $archive"
