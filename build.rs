use std::{
    env, fs,
    io::Cursor,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon.jpg");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let icon_src = manifest_dir.join("assets").join("icon.jpg");
    if !icon_src.exists() {
        println!("cargo:warning=Missing assets/icon.jpg; skipping exe icon.");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ico_path = out_dir.join("app_icon.ico");
    let rc_path = out_dir.join("app_icon.rc");

    if let Err(err) = write_ico(&icon_src, &ico_path) {
        println!("cargo:warning=Failed to build exe icon: {err}");
        return;
    }

    let ico_str = ico_path.to_string_lossy().replace('\\', "/");
    let rc_contents = format!("1 ICON \"{}\"\n", ico_str);
    if fs::write(&rc_path, rc_contents).is_err() {
        println!("cargo:warning=Failed to write icon resource file.");
        return;
    }

    embed_resource::compile(rc_path, embed_resource::NONE);
}

fn write_ico(source: &Path, dest: &Path) -> Result<(), String> {
    let image = image::open(source).map_err(|e| format!("icon load failed: {e}"))?;
    let image = image.resize_exact(256, 256, image::imageops::FilterType::Lanczos3);
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Ico)
        .map_err(|e| format!("icon encode failed: {e}"))?;
    fs::write(dest, bytes).map_err(|e| format!("icon write failed: {e}"))
}
