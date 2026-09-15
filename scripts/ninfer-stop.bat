@echo off
REM ninfer-stop.bat - stop NInfer server, free ALL VRAM
if "%NINFER_PORT%"=="" set NINFER_PORT=11434

echo Stopping ninfer-serve...
taskkill /IM ninfer-serve.exe /F >nul 2>&1
powershell -Command "Get-NetTCPConnection -LocalPort %NINFER_PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"
echo Done. VRAM free for other GPU work.
