cd %GITHUB_WORKSPACE%\openblas\OpenBLAS-0.3.14\build
call "c:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Auxiliary/Build/vcvars64.bat"
set "LIB=%CONDA_PREFIX%/Library/lib;%LIB%"
set "CPATH=%CONDA_PREFIX%/Library/include;%CPATH%"
cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_C_COMPILER=clang-cl -DCMAKE_Fortran_COMPILER=flang -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 -DDYNAMIC_ARCH=ON -DCMAKE_BUILD_TYPE=Release -DUSE_THREAD=ON -DNUM_THREADS=128 -DUSE_OPENMP=0
cmake --build . --config Release
cmake --install . --prefix %GITHUB_WORKSPACE%\openblas-inst -v
