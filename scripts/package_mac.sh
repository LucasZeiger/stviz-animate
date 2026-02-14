#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist="$root/dist/macos"
zip="$root/dist/stviz-animate-mac.zip"

if ! cargo bundle --version >/dev/null 2>&1; then
  echo "cargo-bundle is required. Install with: cargo install cargo-bundle" >&2
  exit 1
fi

cargo build --release
cargo bundle --release

app="$root/target/release/bundle/osx/stviz-animate.app"
if [[ ! -d "$app" ]]; then
  echo "Missing bundle: $app" >&2
  exit 1
fi
if [[ -f "$root/THIRD_PARTY_NOTICES.md" ]]; then
  cp "$root/THIRD_PARTY_NOTICES.md" "$app/Contents/Resources/THIRD_PARTY_NOTICES.md"
fi

rm -rf "$dist"
mkdir -p "$dist"
if [[ -f "$zip" ]]; then
  rm -f "$zip"
fi

ditto -c -k --sequesterRsrc --keepParent "$app" "$zip"
echo "macOS bundle created at $zip"
