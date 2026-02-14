param(
    [string]$OutDir = "dist/windows",
    [switch]$IncludeFfmpeg
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$dist = Join-Path $root $OutDir
$zip = Join-Path $root "dist/stviz-animate-windows.zip"

cargo build --release
if ($LASTEXITCODE -ne 0) {
    throw "cargo build --release failed with exit code $LASTEXITCODE"
}

if (Test-Path $dist) {
    Remove-Item $dist -Recurse -Force
}
New-Item -ItemType Directory -Force $dist | Out-Null

Copy-Item (Join-Path $root "target/release/stviz-animate.exe") $dist
Copy-Item (Join-Path $root "python") -Destination (Join-Path $dist "python") -Recurse
if (Test-Path (Join-Path $root "THIRD_PARTY_NOTICES.md")) {
    Copy-Item (Join-Path $root "THIRD_PARTY_NOTICES.md") $dist
}

if ($IncludeFfmpeg) {
    $ffmpegDir = Join-Path $root "ffmpeg"
    if (Test-Path $ffmpegDir) {
        Copy-Item $ffmpegDir -Destination (Join-Path $dist "ffmpeg") -Recurse
    }
    $ffmpegBin = Join-Path $root "bin/ffmpeg.exe"
    if (Test-Path $ffmpegBin) {
        New-Item -ItemType Directory -Force (Join-Path $dist "bin") | Out-Null
        Copy-Item $ffmpegBin (Join-Path $dist "bin/ffmpeg.exe")
    }
    Write-Host "Included local ffmpeg binaries. Ensure third-party license and patent obligations are satisfied."
}
else {
    Write-Host "Skipping ffmpeg bundling by default. Use -IncludeFfmpeg only if you handle ffmpeg/x264 compliance."
}

if (Test-Path $zip) {
    Remove-Item $zip -Force
}
Compress-Archive -Path (Join-Path $dist "*") -DestinationPath $zip -Force

Write-Host "Windows bundle created at $zip"
