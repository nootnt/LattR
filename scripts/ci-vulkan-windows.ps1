# Installs only the Vulkan SDK parts the build needs (headers, vulkan-1.lib, glslc) into $Dest
# glslc needs the MSVC runtime, which Visual Studio provides
param([string]$Dest = "C:\VulkanSDK\min")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Version = "1.4.363.0"
$Sha = "94a82d378f7a5e3e54c9db7d2fb7016af136e14ac0a18dbf0f2f67a36352d141"

$tmp = [IO.Path]::GetTempPath()
$exe = Join-Path $tmp "vulkansdk-$Version.exe"
$full = Join-Path $tmp "VulkanSDK-$Version"
Invoke-WebRequest "https://sdk.lunarg.com/sdk/download/$Version/windows/vulkansdk-windows-X64-$Version.exe" -OutFile $exe -UseBasicParsing
if ((Get-FileHash $exe -Algorithm SHA256).Hash.ToLower() -ne $Sha) { throw "Vulkan SDK checksum mismatch" }

# GUI-subsystem installer: wait for it explicitly
$p = Start-Process $exe -ArgumentList "--root", $full, "--accept-licenses", "--default-answer", "--confirm-command", "install" -Wait -PassThru -NoNewWindow
if ($p.ExitCode) { throw "Vulkan SDK install failed ($($p.ExitCode))" }

New-Item -ItemType Directory -Force "$Dest\Bin", "$Dest\Lib" | Out-Null
Copy-Item -Recurse -Force "$full\Include" "$Dest\"
Copy-Item "$full\Lib\vulkan-1.lib" "$Dest\Lib\"
Copy-Item "$full\Bin\glslc.exe" "$Dest\Bin\"
Remove-Item -Recurse -Force $full, $exe -ErrorAction SilentlyContinue
Write-Host "Vulkan SDK $Version (minimal) in $Dest"
