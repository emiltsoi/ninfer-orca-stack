@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
cl 2>&1 | findstr /R "Version"
nvcc --version | findstr /R "release"
cmake --version | findstr /R "version"
ninja --version
