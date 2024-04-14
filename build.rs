fn main() {
    println!("cargo:rustc-link-lib=dylib=python3.11");
    println!("cargo:rustc-link-search=native=/Users/riohatta/miniconda3/envs/alpha_zero/lib");
}