@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38 >nul 2>&1
cmake -B build -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES="120a" -DNINFER_ENABLE_AVX2=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
