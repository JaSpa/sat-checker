fn main() {
    cpp_build::build("src/bin/minisat-instance.rs");
    println!("cargo:rerun-if-changed=src/bin/minisat-instance.rs");
}
