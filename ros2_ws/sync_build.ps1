# 一键：把 Windows 侧的 ROS2 包同步到 WSL 并构建
# 用法（PowerShell）：powershell -ExecutionPolicy Bypass -File ros2_ws\sync_build.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$wslPath = "/mnt/" + ($repo.Substring(0, 1).ToLower()) + ($repo.Substring(2) -replace '\\', '/')
Write-Host "repo in WSL: $wslPath/ros2_ws"
wsl.exe -d Ubuntu-22.04 -- bash -lc "bash '$wslPath/ros2_ws/build.sh'"
if ($LASTEXITCODE -ne 0) { throw "WSL 构建失败（exit $LASTEXITCODE）" }
Write-Host "构建成功：~/scara_ros2_ws/install"
