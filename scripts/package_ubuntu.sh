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
if [[ -f "$root/THIRD_PARTY_NOTICES.md" ]]; then
  cp "$root/THIRD_PARTY_NOTICES.md" "$dist/THIRD_PARTY_NOTICES.md"
fi

if [[ "${INCLUDE_FFMPEG:-0}" == "1" ]]; then
  if [[ -d "$root/ffmpeg" ]]; then
    cp -R "$root/ffmpeg" "$dist/ffmpeg"
  fi
  if [[ -f "$root/bin/ffmpeg" ]]; then
    mkdir -p "$dist/bin"
    cp "$root/bin/ffmpeg" "$dist/bin/ffmpeg"
  fi
  echo "Included local ffmpeg binaries. Ensure third-party license and patent obligations are satisfied."
else
  echo "Skipping ffmpeg bundling by default. Set INCLUDE_FFMPEG=1 only if you handle ffmpeg/x264 compliance."
fi

mkdir -p "$root/dist"
tar -czf "$archive" -C "$root/dist/linux" "stviz-animate"
echo "Ubuntu bundle created at $archive"
