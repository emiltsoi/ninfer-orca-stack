@echo off
REM ninfer-ctl-stop.bat - stop ninfer-ctl AND its ninfer-serve child (free ALL VRAM)

if "%NINFER_PORT%"=="" set NINFER_PORT=11434
if "%NINFER_CHILD_PORT%"=="" set NINFER_CHILD_PORT=11435

echo Stopping ninfer-serve child(ren)...
taskkill /IM ninfer-serve.exe /F >nul 2>&1
powershell -Command "Get-NetTCPConnection -LocalPort %NINFER_CHILD_PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"

echo Stopping ninfer-ctl controller (by cmdline match)...
powershell -Command "Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" | Where-Object { $_.CommandLine -like '*ninfer-ctl.js*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"

REM Fallback: whoever owns the public port
powershell -Command "Get-NetTCPConnection -LocalPort %NINFER_PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"

echo Done. VRAM should be fully free.
