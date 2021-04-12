:: This builds openblas on windows, and is a copy of the instructions used for
:: building OpenBLAS on the Windows CI.
:: Set the environment variable GITHUB_WORKSPACE accordingly.
:: Build of OpenBLAS on other platforms is handled by the openblas-src rust
:: crate (https://crates.io/crates/openblas-src).
:: Requires openblas 0.3.14 to be installed and untar-ed in ./openblas/
:: Requires Visual Studio 2019 to be installed (change Enterprise to Community if needed).
:: Requires perl, ninja, cmake, clang-cl and flang to be installed. This can be done with conda:
::     conda install -y cmake flang clangdev perl libflang ninja
cd %GITHUB_WORKSPACE%\openblas\OpenBLAS-0.3.14\build
call "c:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Auxiliary/Build/vcvars64.bat"
cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_C_COMPILER=clang-cl -DCMAKE_Fortran_COMPILER=flang -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 -DDYNAMIC_ARCH=ON -DCMAKE_BUILD_TYPE=Release -DUSE_THREAD=ON -DNUM_THREADS=128 -DUSE_OPENMP=0 -DFCOMMON_OPT=-static-flang-libs
cmake --build . --config Release
cmake --install . --prefix %GITHUB_WORKSPACE%\openblas\inst -v
