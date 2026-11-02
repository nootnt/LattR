# Builds dist\LattR-<version>-windows-x64.zip with LattR.exe, license notices and portable.txt
# Needs Rust (MSVC), Visual Studio C++ Build Tools, CMake, LLVM and the Vulkan SDK
param(
    [switch]$CpuOnly
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location (Join-Path $PSScriptRoot "..")

# keep the nested whisper.cpp Vulkan build under Windows' 260-character path limit
if (-not $env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR = "$env:SystemDrive\lattr-target" }
$target = $env:CARGO_TARGET_DIR

$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches[0].Groups[1].Value
$name = "LattR-$version-windows-x64"
$dist = "dist\$name"

foreach ($cmd in "cargo", "cmake") {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "$cmd not found in PATH" }
}
if (-not $env:LIBCLANG_PATH) {
    $llvm = "$env:ProgramFiles\LLVM\bin"
    if (-not (Test-Path "$llvm\libclang.dll")) { throw "libclang not found: install LLVM or set LIBCLANG_PATH" }
    $env:LIBCLANG_PATH = $llvm
}
if (-not $CpuOnly -and -not $env:VULKAN_SDK) { throw "VULKAN_SDK is not set: install the Vulkan SDK or pass -CpuOnly" }

$features = if ($CpuOnly) { @("--no-default-features") } else { @() }
cargo build --release --locked @features
if ($LASTEXITCODE) { throw "cargo build failed" }

if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force $dist | Out-Null
Copy-Item "$target\release\lattr.exe" "$dist\LattR.exe"
Copy-Item README.md "$dist\README.md"
Copy-Item LICENSE "$dist\LICENSE"
Copy-Item THIRD_PARTY_LICENSES.html "$dist\THIRD_PARTY_LICENSES.html"
Set-Content "$dist\portable.txt" "Settings, models and caches are stored in the data folder next to LattR.exe.`r`nDelete this file to use your user profile (AppData) instead."

$zipOut = "dist\$name.zip"
if (Test-Path $zipOut) { Remove-Item $zipOut }
Compress-Archive -Path $dist -DestinationPath $zipOut -CompressionLevel Optimal
Write-Host "built $zipOut"
