fn main() {
    cpp_build::build("src/main.rs");
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rustc-link-lib=minisat");
}
