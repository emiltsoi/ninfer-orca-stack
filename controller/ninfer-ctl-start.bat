@echo off
REM ninfer-ctl-start.bat - start the NInfer model controller (manual, on-demand)
REM Survives SSH session close (uses WMI detach, not start /b).
REM The controller itself is tiny (no VRAM). Profiles load lazily on first API
REM call (~6s boot) and stay loaded until: another alias is requested,
REM POST /yield, or ninfer-ctl-stop.bat is run.
REM
REM Configure via environment: NINFER_HOME (engine checkout), NINFER_PORT.

cd /d %~dp0

if "%NINFER_PORT%"=="" set NINFER_PORT=11434
powershell -Command "if (Get-NetTCPConnection -LocalPort %NINFER_PORT% -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
  echo Port %NINFER_PORT% already in use - ninfer-ctl or a direct ninfer-serve is already running.
  goto :eof
)

echo Starting ninfer-ctl (detached)...
powershell -Command "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c node %~dp0ninfer-ctl.js > %~dp0ninfer-ctl.log 2>&1'}; if ($r.ReturnValue -eq 0) { Write-Output ('started PID ' + $r.ProcessId) } else { Write-Output ('FAILED ' + $r.ReturnValue) }"
echo Health: http://<your-host>:%NINFER_PORT%/health
echo Models: qwen-3.8-orca, qwen-3.8-orca-fast, qwen-3.8-orca-vision, qwen-3.8-orca-vision-fast
