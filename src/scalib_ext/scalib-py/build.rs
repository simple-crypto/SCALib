fn main() {
    if cfg!(target_os = "windows") {
        // We require a MSVC-compatible static build of OpenBLAS with LAPACK on Windows.
        //
        // Example build instructions (adapted from
        // https://github.com/xianyi/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio,
        // as of 2021-04-07).
        //
        // 1. Download OpenBLAS from https://github.com/xianyi/OpenBLAS/releases (tested 0.3.14)
        //    and untar:
        //    ```
        //    mkdir openblas
        //    cd openblas
        //    curl -L -O https://github.com/xianyi/OpenBLAS/releases/download/v0.3.14/OpenBLAS-0.3.14.tar.gz
        //    tar xzf OpenBLAS-0.3.14.tar.gz
        //    cd OpenBLAS-0.3.14
        //    ```
        // 2. Install Miniconda3 for 64 bits. Run the following in the miniconda prompt.
        // 3. Install compilers and other build dependencies from miniconda (flang is to old,
        //    non-F18, LLVM fortran compiler):
        //    ```
        //    conda update -n base conda
        //    conda config --add channels conda-forge
        //    conda install -y cmake flang clangdev perl libflang ninja
        //    ```
        // 4. Activate 64-bit MSVC environment (requries Visual Studio 2019 to be installed), adapt
        //    community to Enterprise if needed:
        //    ```
        //    "c:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat"
        //    ```
        // 5. Configure CMake:
        //    ```
        //    set "LIB=%CONDA_PREFIX%\Library\lib;%LIB%"
        //    set "CPATH=%CONDA_PREFIX%\Library\include;%CPATH%"
        //    mkdir build
        //    cd build
        //    cmake .. -G "Ninja" -DCMAKE_CXX_COMPILER=clang-cl -DCMAKE_C_COMPILER=clang-cl \
        //    -DCMAKE_Fortran_COMPILER=flang -DBUILD_WITHOUT_LAPACK=no -DNOFORTRAN=0 \
        //    -DDYNAMIC_ARCH=ON -DCMAKE_BUILD_TYPE=Release -DUSE_THREAD=ON -DNUM_THREADS=128 \
        //    -DUSE_OPENMP=0
        //    ```
        // 6. Build the project:
        //    ```
        //    cmake --build . --config Release
        //    ```
        // 7. Install in a chosen dir (OPENBLAS_INSTALL_DIR):
        //    ```
        //    cmake --install . --prefix %OPENBLAS_INSTALL_DIR% -v
        //    ```
        // 8. The location of the OpenBLAS library is
        //    SCALIB_OPENBLAS_LIB_DIR=%OPENBLAS_INSTALL_DIR%\lib\openblas.lib
        //
        let openblas_lib_env_name = "SCALIB_OPENBLAS_LIB_DIR";
        let openblas_lib = std::env::var_os(openblas_lib_env_name)
            .expect(&format!("{} not defined.", openblas_lib_env_name));
        let path = std::path::Path::new(&openblas_lib);
        let mut path_lib = path.to_owned();
        path_lib.push("openblas.lib");
        if !path.exists() {
            panic!(
                "No file at {} (computed from {})",
                path_lib.to_string_lossy(),
                openblas_lib_env_name
            );
        }
        println!("cargo:rustc-link-search=native={}", path.to_string_lossy());
        println!("cargo:rustc-link-lib=static=openblas");
        println!("cargo:rerun-if-env-changed={}", openblas_lib_env_name);
        println!("cargo:rerun-if-changed={}", path_lib.to_string_lossy());
    } else {
        // Do not re-run.
        println!("cargo:rerun-if-changed=build.rs");
    }
}
