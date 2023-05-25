fn main() {
    cpp_build::Config::new()
        .flag("-std=c++17")
        .build("src/main.rs");
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rustc-link-lib=z");
    println!("cargo:rustc-link-lib=minisat");
}
