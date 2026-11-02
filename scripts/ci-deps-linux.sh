#!/usr/bin/env bash

# Installs CI build dependencies in an Ubuntu 22.04 container, run as root
# Vulkan headers and glslc come from the LunarG SDK into $1 (default .ci/vulkan-sdk)
# Exports VULKAN_SDK and PATH via $GITHUB_ENV / $GITHUB_PATH
set -euo pipefail

SDK_VERSION=1.4.363.0
SDK_SHA=197962f5cbf80baf2775a03336a01cee7c8745686c65aaa70d3f751ade4d7e43
sdk=$(realpath -m "${1:-.ci/vulkan-sdk}")

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    build-essential cmake clang libclang-dev pkg-config git curl ca-certificates xz-utils file \
    libasound2-dev libvulkan-dev >/dev/null

if [[ ! -x $sdk/bin/glslc ]]; then
    rm -rf "$sdk" /tmp/vksdk
    mkdir -p /tmp/vksdk
    curl -fsSL -o /tmp/vksdk.tar.xz "https://sdk.lunarg.com/sdk/download/$SDK_VERSION/linux/vulkansdk-linux-x86_64-$SDK_VERSION.tar.xz"
    echo "$SDK_SHA  /tmp/vksdk.tar.xz" | sha256sum -c --quiet
    # only glslc and the headers are needed
    tar -xJf /tmp/vksdk.tar.xz -C /tmp/vksdk --no-same-owner --wildcards '*/x86_64/bin/glslc' '*/x86_64/include/vulkan/*' '*/x86_64/include/vk_video/*'
    rm /tmp/vksdk.tar.xz
    mkdir -p "$(dirname "$sdk")"
    mv "/tmp/vksdk/$SDK_VERSION/x86_64" "$sdk"
fi
# link against the distro's (older) loader so newer-only symbols fail at build time
mkdir -p "$sdk/lib"
ln -sf "$(ldconfig -p | awk '/libvulkan\.so\.1 .*x86-64/ {print $NF; exit}')" "$sdk/lib/libvulkan.so"

if ! command -v cargo >/dev/null; then
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null
    echo "$HOME/.cargo/bin" >> "${GITHUB_PATH:-/dev/null}"
fi
echo "VULKAN_SDK=$sdk" >> "${GITHUB_ENV:-/dev/null}"
echo "VULKAN_SDK=$sdk"
