@echo off
REM ninfer-start.bat - single-profile NInfer server (orca NVFP4: MTP, vision, YaRN 288k)
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

echo Starting ninfer-serve (detached)...
powershell -Command "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c %NINFER_HOME%\build\apps\ninfer-serve.exe %NINFER_HOME%\models\qwen3_8_27b_orca_nvfp4.ninfer --host 0.0.0.0 --port %NINFER_PORT% --vision --max-context 288000 --kv-capacity auto --max-concurrency 1 --kv-dtype nvfp4 --spec mtp --draft-tokens 5 --lm-head-draft --rope-yarn-factor 1.25 --rope-original-max-position 262144 --cors > %NINFER_HOME%\ninfer-serve.log 2>&1'}; if ($r.ReturnValue -eq 0) { Write-Output ('started PID ' + $r.ProcessId) } else { Write-Output ('FAILED ' + $r.ReturnValue) }"
echo Health: http://<your-host>:%NINFER_PORT%/health
echo Model: qwen3.8-27b (orca uncensored, vision enabled, 288k ctx)
