copy %CONDA_PREFIX%\Library\lib\flangmain.lib %GITHUB_WORKSPACE%\openblas\flib\
copy %CONDA_PREFIX%\Library\lib\libflang.lib %GITHUB_WORKSPACE%\openblas\flib\
copy %CONDA_PREFIX%\Library\lib\libflangrti.lib %GITHUB_WORKSPACE%\openblas\flib\
copy %CONDA_PREFIX%\Library\lib\libompstub.lib %GITHUB_WORKSPACE%\openblas\flib\
copy %CONDA_PREFIX%\Library\lib\libpgmath.lib %GITHUB_WORKSPACE%\openblas\flib\
cd %GITHUB_WORKSPACE%\openblas\OpenBLAS-0.3.14\build
call "c:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Auxiliary/Build/vcvars64.bat"

set "LIB=%GITHUB_WORKSPACE%/openblas/flib;%CONDA_PREFIX%/Library/lib;%LIB%"
set "CPATH=%CONDA_PREFIX%/Library/include;%CPATH%"
cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_C_COMPILER=clang-cl -DCMAKE_Fortran_COMPILER=flang -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 -DDYNAMIC_ARCH=ON -DCMAKE_BUILD_TYPE=Release -DUSE_THREAD=ON -DNUM_THREADS=128 -DUSE_OPENMP=0 -DFCOMMON_OPT=-static-flang-libs
cmake --build . --config Release
cmake --install . --prefix %GITHUB_WORKSPACE%\openblas\inst -v
