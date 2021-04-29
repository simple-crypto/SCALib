fn main() {
    //cxx_build::CFG.include_prefix = "geigen/include";
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR").unwrap();
    let include_dir = std::path::Path::new(&manifest_dir).join("include");
    cxx_build::bridge("src/geigen.rs")
        .file("src/geigen.cpp")
        .include(&include_dir)
        .define("EIGEN_MPL2_ONLY", None)
        .flag_if_supported("-std=c++14")
        .compile("cxxgeigen");
    println!("cargo:rerun-if-changed=src/geigen.rs");
    println!("cargo:rerun-if-changed=src/geigen.cpp");
    println!("cargo:rerun-if-changed=include/geigen.h");
}
