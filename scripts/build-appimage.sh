#!/usr/bin/env bash

# Builds dist/LattR-<version>-x86_64.AppImage with the app
# glibc, libstdc++, ALSA, Vulkan and ffmpeg come from the host system
#
# Env: CPU_ONLY=1 builds without Vulkan
set -euo pipefail
cd "$(dirname "$0")/.."

APPIMAGETOOL_URL=https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage
APPIMAGETOOL_SHA=ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0
RUNTIME_URL=https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64
RUNTIME_SHA=2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d

version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
tools=target/appimage-tools
appdir=target/AppDir
out=dist/LattR-$version-x86_64.AppImage
mkdir -p "$tools" dist

# download $1 to $2 unless it already exists with sha256 $3
fetch() {
    if [[ -f $2 ]] && echo "$3  $2" | sha256sum -c --status; then return; fi
    curl -fL --retry 3 -o "$2.part" "$1"
    echo "$3  $2.part" | sha256sum -c --quiet || { rm -f "$2.part"; echo "checksum mismatch for $1" >&2; exit 1; }
    mv "$2.part" "$2"
}

features=()
[[ -n ${CPU_ONLY:-} ]] && features=(--no-default-features)
cargo build --release --locked "${features[@]}"

fetch "$APPIMAGETOOL_URL" "$tools/appimagetool" "$APPIMAGETOOL_SHA"
fetch "$RUNTIME_URL" "$tools/runtime" "$RUNTIME_SHA"
chmod +x "$tools/appimagetool"

rm -rf "$appdir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/share/icons/hicolor/256x256/apps" "$appdir/usr/share/applications" "$appdir/usr/share/doc/lattr"
install -m755 target/release/lattr "$appdir/usr/bin/lattr"
install -m644 packaging/lattr.desktop "$appdir/usr/share/applications/lattr.desktop"
install -m644 assets/icon.png "$appdir/usr/share/icons/hicolor/256x256/apps/lattr.png"
install -m644 README.md "$appdir/usr/share/doc/lattr/README.md"
install -m644 LICENSE "$appdir/usr/share/doc/lattr/LICENSE"
install -m644 THIRD_PARTY_LICENSES.html "$appdir/usr/share/doc/lattr/THIRD_PARTY_LICENSES.html"
if [[ -z ${CPU_ONLY:-} ]]; then
    # fallback Vulkan loader, still uses the system GPU drivers
    mkdir -p "$appdir/usr/lib/fallback"
    cp -L "$(ldconfig -p | awk '/libvulkan\.so\.1 .*x86-64/ {print $NF; exit}')" "$appdir/usr/lib/fallback/libvulkan.so.1"
fi
ln -s usr/share/applications/lattr.desktop "$appdir/lattr.desktop"
ln -s usr/share/icons/hicolor/256x256/apps/lattr.png "$appdir/lattr.png"
ln -s lattr.png "$appdir/.DirIcon"
cat > "$appdir/AppRun" <<'EOF'
#!/bin/sh
HERE=${APPDIR:-$(dirname "$(readlink -f "$0")")}
if ! { /sbin/ldconfig -p 2>/dev/null || ldconfig -p 2>/dev/null; } | grep -q 'libvulkan\.so\.1'; then
    export LD_LIBRARY_PATH="$HERE/usr/lib/fallback${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$HERE/usr/bin/lattr" "$@"
EOF
chmod +x "$appdir/AppRun"

# runs without FUSE, so this also works in containers and CI
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$tools/appimagetool" --no-appstream --runtime-file "$tools/runtime" "$appdir" "$out"

glibc=$(objdump -T target/release/lattr | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
echo "built $out (needs $glibc or newer)"
