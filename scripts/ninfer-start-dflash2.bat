@echo off
REM ninfer-start-dflash2.bat - single-profile NInfer server (orca NVFP4 + DFlash2 draft: text + vision, 200k)
REM Detached via WMI - survives console/SSH session close.
REM Requires NINFER_HOME pointing at the ninfer-5090-windows checkout.
if "%NINFER_HOME%"=="" set NINFER_HOME=%~dp0..\ninfer-5090-windows
if "%NINFER_PORT%"=="" set NINFER_PORT=11434
cd /d %NINFER_HOME%

powershell -Command "if (Get-NetTCPConnection -LocalPort %NINFER_PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
  echo Port %NINFER_PORT% already in use - ninfer-serve or another server is already running.
  goto :eof
)

echo Starting ninfer-serve DFlash2 (detached)...
powershell -Command "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c %NINFER_HOME%\build\apps\ninfer-serve.exe %NINFER_HOME%\models\qwen3_8_27b_orca_nvfp4_dflash2.ninfer --host 0.0.0.0 --port %NINFER_PORT% --vision --max-context 200000 --kv-capacity auto --max-concurrency 1 --kv-dtype nvfp4 --spec dflash2 --draft-tokens 7 --lm-head-draft --cors > %NINFER_HOME%\ninfer-serve.log 2>&1'}; if ($r.ReturnValue -eq 0) { Write-Output ('started PID ' + $r.ProcessId) } else { Write-Output ('FAILED ' + $r.ReturnValue) }"
echo Health: http://<your-host>:%NINFER_PORT%/health
echo Model: qwen3.8-27b (orca uncensored, vision enabled, 200k ctx, DFlash2)
