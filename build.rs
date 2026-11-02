fn main() {
    println!("cargo:rerun-if-changed=cheetah_init.c");
    println!("cargo:rerun-if-changed=cheetah_init.h");
    println!("cargo:rerun-if-changed=cheetah_vulkan.c");
    println!("cargo:rerun-if-changed=cheetah_vulkan.h");

    cc::Build::new()
        .file("cheetah_init.c")
        .include(".")
        .warnings(true)
        .extra_warnings(true)
        .compile("cheetah_init");

    if std::env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        cc::Build::new()
            .file("cheetah_vulkan.c")
            .include(".")
            .warnings(true)
            .extra_warnings(true)
            .compile("cheetah_vulkan");
        println!("cargo:rustc-link-lib=vulkan");
    }
}
