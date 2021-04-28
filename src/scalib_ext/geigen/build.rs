fn main() {
    cxx_build::bridge("src/geigen.rs")
        .file("src/geigen.cpp")
        .flag_if_supported("-std=c++14")
        .compile("cxxgeigen");
    println!("cargo:rerun-if-changed=src/geigen.rs");
    println!("cargo:rerun-if-changed=src/geigen.cpp");
    println!("cargo:rerun-if-changed=include/geigen.h");
}
