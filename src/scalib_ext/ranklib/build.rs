use std::env;
use std::path::PathBuf;

fn gen_binding_bignum() {
    println!("cargo:rerun-if-changed=src/bignumpoly.h");
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/bignumpoly.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .size_t_is_usize(true)
        .prepend_enum_name(false)
        .raw_line("#[repr(C)] pub struct NTL_ZZ { _private: [u64; 1]}")
        .raw_line("#[repr(C)] pub struct NTL_ZZX { _private: [u64; 1]}")
        .allowlist_function("bnp_.*")
        .allowlist_recursively(false)
        .clang_args(vec!["-x", "c++"])
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("binding_bnp.rs"))
        .expect("Couldn't write bindings!");
}

fn gen_bindings() {
    println!("cargo:rerun-if-changed=src/hel_if.h");
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/hel_if.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .size_t_is_usize(true)
        .prepend_enum_name(false)
        .raw_line("#[repr(C)] pub struct NTL_ZZ { _private: [u64; 1]}")
        .raw_line("#[repr(C)] pub struct NTL_ZZX { _private: [u64; 1]}")
        .raw_line("#[repr(C)] pub struct NTL_RR { _private: [u64; 2]}")
        .allowlist_type("hel_.*")
        .allowlist_function("hel_.*")
        .allowlist_function("helc_.*")
        .allowlist_recursively(false)
        .clang_args(vec!["-x", "c++"])
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("binding_hel_if.rs"))
        .expect("Couldn't write bindings!");
}

fn build_bignumpoly() {
    println!("cargo:rerun-if-changed=src/bignumpoly.cpp");
    println!("cargo:rerun-if-changed=src/bignumpoly.h");
    cc::Build::new()
        .cpp(true)
        .warnings(false)
        .flag("-O3")
        .file("src/bignumpoly.cpp")
        .compile("bignumpoly");
    println!("cargo:rustc-link-lib=ntl");
    println!("cargo:rustc-link-lib=gmp");
}

fn build_c_lib() {
    let cpp_files = vec![
        "src/aes.cpp",
        "src/hel_enum.cpp",
        "src/hel_execute.cpp",
        "src/hel_histo.cpp",
        "src/hel_init.cpp",
        "src/hel_util.cpp",
        "src/scores_example.cpp",
        "src/scores_example_data.cpp",
        "src/top.cpp",
        "src/hel_if.cpp",
    ];
    let mut hellib = cc::Build::new();
    hellib.cpp(true);
    hellib.warnings(false);
    hellib.flag("-O3");
    for f in &cpp_files {
        hellib.file(f);
        println!("cargo:rerun-if-changed={}", f);
    }
    hellib.compile("hellib");

    println!("cargo:rustc-link-lib=ntl");
    println!("cargo:rustc-link-lib=gmp");
}

fn main() {
    if std::env::var("CARGO_FEATURE_HELLIB").is_ok() {
        gen_bindings();
        build_c_lib();
    }
    if std::env::var("CARGO_FEATURE_NTL").is_ok() {
        gen_binding_bignum();
        build_bignumpoly();
    }
}
