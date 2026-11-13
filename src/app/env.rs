//! Locating Python, ffmpeg and the app/project directories.

use super::{util::chrono_like_timestamp, StvizApp, APP_DATA_DIR_NAME};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

impl StvizApp {
    pub(super) fn detect_python_cmd() -> Result<(String, String), String> {
        let mut errors = Vec::new();
        if let Some(env_cmd) = Self::python_from_env() {
            match Self::check_python_cmd(&env_cmd) {
                Ok(result) => return Ok(result),
                Err(msg) => errors.push(msg),
            }
        }

        // `py` is the python.org launcher, present even when Python was
        // installed without being added to PATH.
        let candidates: &[&str] = if cfg!(windows) {
            &["python", "py", "python3"]
        } else {
            &["python3", "python"]
        };
        // The Microsoft Store Python redirects writes under AppData to a
        // private folder, so the app cannot find the environment it creates
        // there. Use it only if nothing else is installed.
        let mut store_python = None;
        for &candidate in candidates {
            match Self::check_python_cmd(candidate) {
                Ok(result) if is_store_python(&result.1) => {
                    store_python.get_or_insert(result);
                }
                Ok(result) => return Ok(result),
                Err(msg) => errors.push(msg),
            }
        }
        // Standard install folders, so Python is found even when it isn't on
        // PATH, and right after the app installed it (PATH changes only reach
        // newly started programs).
        if cfg!(windows) {
            let local = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
            let program_files = std::env::var_os("ProgramFiles").map(PathBuf::from);
            for exe in windows_python_locations(local.as_deref(), program_files.as_deref()) {
                let exe = exe.display().to_string();
                match Self::check_python_cmd(&exe) {
                    Ok(result) => return Ok(result),
                    Err(msg) => errors.push(msg),
                }
            }
        }
        if let Some(result) = store_python {
            return Ok(result);
        }
        let hint = if cfg!(windows) {
            "Use Install Python below, or install Python 3.9 or newer from python.org."
        } else {
            "Install Python 3.9 or newer (e.g. `sudo apt install python3 python3-venv`, or \
             python.org on macOS), then restart the app."
        };
        if errors.is_empty() {
            Err(format!("Python not found.\n{hint}"))
        } else {
            Err(format!("{}\n\n{hint}", errors.join("\n")))
        }
    }

    fn check_python_cmd(cmd: &str) -> Result<(String, String), String> {
        let env_root = Self::python_env_root(cmd);
        let mut exe_cmd = Command::new(cmd);
        exe_cmd
            .arg("-c")
            .arg(
                "import sys; print(sys.executable); print(f\"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}\")",
            );
        Self::apply_python_env(&mut exe_cmd, env_root.as_deref());
        let exe_out = exe_cmd
            .output()
            .map_err(|e| format!("`{cmd}` not found: {e}"))?;
        if !exe_out.status.success() {
            return Err(format!("`{cmd}` failed to run (exit {}).", exe_out.status));
        }
        let stdout = String::from_utf8_lossy(&exe_out.stdout);
        let mut lines = stdout.lines();
        let exe = lines.next().unwrap_or("").trim().to_string();
        let version = lines.next().unwrap_or("").trim().to_string();
        if !version.is_empty() {
            let mut parts = version.split(['.', ' ']);
            let major = parts
                .next()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0);
            let minor = parts
                .next()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0);
            if major > 0 && (major, minor) < MIN_PYTHON_VERSION {
                return Err(format!(
                    "`{cmd}` is Python {version}, but converting needs Python {}.{} or newer. \
                     Install a newer Python and restart the app.",
                    MIN_PYTHON_VERSION.0, MIN_PYTHON_VERSION.1
                ));
            }
        }
        let exe_label = if exe.is_empty() { cmd.to_string() } else { exe };

        Ok((cmd.to_string(), exe_label))
    }

    fn python_from_env() -> Option<String> {
        let env_prefix =
            std::env::var_os("VIRTUAL_ENV").or_else(|| std::env::var_os("CONDA_PREFIX"))?;
        let base = PathBuf::from(env_prefix);
        let candidates = if cfg!(windows) {
            ["Scripts/python.exe", "python.exe"]
        } else {
            ["bin/python", "bin/python3"]
        };
        for rel in candidates {
            let path = base.join(rel);
            if path.exists() {
                return Some(path.display().to_string());
            }
        }
        None
    }

    pub(super) fn resolve_app_data_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("STVIZ_DATA_DIR") {
            let trimmed = dir.trim();
            if !trimmed.is_empty() {
                return PathBuf::from(trimmed);
            }
        }
        #[cfg(target_os = "windows")]
        if let Some(dir) = std::env::var_os("APPDATA") {
            return PathBuf::from(dir).join(APP_DATA_DIR_NAME);
        }
        #[cfg(target_os = "macos")]
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join(APP_DATA_DIR_NAME);
        }
        if let Some(dir) = std::env::var_os("XDG_DATA_HOME") {
            return PathBuf::from(dir).join(APP_DATA_DIR_NAME);
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join(APP_DATA_DIR_NAME);
        }
        std::env::temp_dir().join(APP_DATA_DIR_NAME)
    }

    pub(super) fn resolve_project_dir() -> PathBuf {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        if cwd.join("python").join("export_stviz.py").exists() {
            return cwd;
        }
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(mut dir) = exe_path.parent().map(|p| p.to_path_buf()) {
                #[cfg(target_os = "macos")]
                if let Some(contents_dir) = dir.parent() {
                    let resources_dir = contents_dir.join("Resources");
                    if resources_dir
                        .join("python")
                        .join("export_stviz.py")
                        .exists()
                    {
                        return resources_dir;
                    }
                }
                if let Some(parent) = dir.parent() {
                    let share_dir = parent.join("share").join("stviz-animate");
                    if share_dir.join("python").join("export_stviz.py").exists() {
                        return share_dir;
                    }
                }
                for _ in 0..6 {
                    if dir.join("python").join("export_stviz.py").exists() {
                        return dir;
                    }
                    let Some(parent) = dir.parent() else {
                        break;
                    };
                    dir = parent.to_path_buf();
                }
            }
        }
        cwd
    }

    pub(super) fn managed_export_output_path(&self) -> PathBuf {
        let raw = self.export_name.trim();
        // Keep only the last path component, whichever separator was typed.
        let raw = raw.rsplit(['/', '\\']).next().unwrap_or(raw);
        let raw = sanitize_file_name(raw, "");
        let raw = raw.as_str();
        let mut file_name = if raw.is_empty() {
            String::from("stviz-animate_loop.mp4")
        } else {
            Path::new(raw)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("stviz-animate_loop.mp4")
                .trim()
                .to_string()
        };
        if file_name.is_empty() {
            file_name = String::from("stviz-animate_loop.mp4");
        }
        if Path::new(&file_name).extension().is_none() {
            file_name.push_str(".mp4");
        }
        self.output_dir.join(file_name)
    }

    pub(super) fn managed_export_temp_path(&self, final_out: &Path) -> PathBuf {
        let stem = final_out
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("stviz-animate_loop");
        let ext = final_out
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("mp4");
        self.export_dir.join(format!(
            "{stem}.partial.{}.{}",
            chrono_like_timestamp(),
            ext
        ))
    }

    fn canonicalize_for_guard(path: &Path) -> Option<PathBuf> {
        if path.exists() {
            return std::fs::canonicalize(path).ok();
        }
        let parent = path.parent().and_then(|p| std::fs::canonicalize(p).ok())?;
        let name = path.file_name()?;
        Some(parent.join(name))
    }

    pub(super) fn path_within_root(root: &Path, path: &Path) -> bool {
        let root = Self::canonicalize_for_guard(root).unwrap_or_else(|| root.to_path_buf());
        let Some(path) = Self::canonicalize_for_guard(path) else {
            return false;
        };
        path.starts_with(root)
    }

    pub(super) fn resolve_ffmpeg_path(project_dir: &Path) -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(env) = std::env::var("STVIZ_FFMPEG") {
            if !env.trim().is_empty() {
                candidates.push(PathBuf::from(env));
            }
        }
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(dir) = exe_path.parent() {
                candidates.push(dir.join("ffmpeg"));
                candidates.push(dir.join("ffmpeg.exe"));
                candidates.push(dir.join("bin").join("ffmpeg"));
                candidates.push(dir.join("bin").join("ffmpeg.exe"));
            }
        }
        candidates.push(project_dir.join("ffmpeg"));
        candidates.push(project_dir.join("ffmpeg.exe"));
        candidates.push(project_dir.join("bin").join("ffmpeg"));
        candidates.push(project_dir.join("bin").join("ffmpeg.exe"));
        candidates.push(project_dir.join("ffmpeg").join("ffmpeg"));
        candidates.push(project_dir.join("ffmpeg").join("ffmpeg.exe"));
        candidates.push(project_dir.join("ffmpeg").join("bin").join("ffmpeg"));
        candidates.push(project_dir.join("ffmpeg").join("bin").join("ffmpeg.exe"));
        // Where winget and Chocolatey put ffmpeg. Their PATH changes only reach
        // newly started programs, so look there directly after an install.
        if cfg!(windows) {
            if let Some(local) = std::env::var_os("LOCALAPPDATA") {
                let winget = PathBuf::from(local).join("Microsoft").join("WinGet");
                candidates.push(winget.join("Links").join("ffmpeg.exe"));
                if let Some(path) = find_winget_ffmpeg(&winget.join("Packages")) {
                    candidates.push(path);
                }
            }
            let choco = std::env::var_os("ChocolateyInstall")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData\chocolatey"));
            candidates.push(choco.join("bin").join("ffmpeg.exe"));
        }

        for candidate in candidates {
            if candidate.exists() {
                let mut cmd = Command::new(&candidate);
                Self::apply_subprocess_flags(&mut cmd);
                if cmd.arg("-version").output().is_ok() {
                    return Some(candidate);
                }
            }
        }

        let mut cmd = Command::new("ffmpeg");
        Self::apply_subprocess_flags(&mut cmd);
        if cmd.arg("-version").output().is_ok() {
            return Some(PathBuf::from("ffmpeg"));
        }
        None
    }

    pub(super) fn refresh_ffmpeg_path(&mut self) {
        let path = Self::resolve_ffmpeg_path(&self.project_dir);
        self.ffmpeg_available = path.is_some();
        self.ffmpeg_path = path;
    }

    pub(super) fn apply_python_env(cmd: &mut Command, env_root: Option<&Path>) -> Option<String> {
        Self::apply_subprocess_flags(cmd);
        // Piped output on Windows defaults to the legacy code page, where
        // printing a path with non-ASCII characters raises an error.
        cmd.env("PYTHONIOENCODING", "utf-8");
        cmd.env("PYTHONUTF8", "1");
        let env_root = env_root?;

        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
        cmd.env_remove("CONDA_PREFIX");
        cmd.env_remove("CONDA_DEFAULT_ENV");
        cmd.env_remove("CONDA_SHLVL");
        cmd.env_remove("VIRTUAL_ENV");

        if env_root.join("conda-meta").exists() {
            if let Some(name) = env_root.file_name().and_then(|s| s.to_str()) {
                cmd.env("CONDA_DEFAULT_ENV", name);
            }
            cmd.env("CONDA_PREFIX", env_root);
            cmd.env("CONDA_SHLVL", "1");
            cmd.env("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", "1");
        } else if env_root.join("pyvenv.cfg").exists() {
            cmd.env("VIRTUAL_ENV", env_root);
        }

        let path_sep = if cfg!(windows) { ";" } else { ":" };
        let mut parts = Vec::new();
        if cfg!(windows) {
            parts.push(env_root.join("Library").join("bin"));
            parts.push(env_root.join("Scripts"));
        } else {
            parts.push(env_root.join("bin"));
        }
        parts.push(env_root.to_path_buf());
        let prefix = parts
            .iter()
            .filter_map(|p| p.to_str())
            .collect::<Vec<_>>()
            .join(path_sep);
        if let Some(old_path) = std::env::var_os("PATH") {
            let old_path = old_path.to_string_lossy();
            let new_path = if prefix.is_empty() {
                old_path.to_string()
            } else {
                format!("{}{}{}", prefix.as_str(), path_sep, old_path)
            };
            cmd.env("PATH", new_path);
        } else if !prefix.is_empty() {
            cmd.env("PATH", &prefix);
        }
        Some(prefix)
    }

    pub(super) fn apply_subprocess_flags(_cmd: &mut Command) {
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            _cmd.creation_flags(CREATE_NO_WINDOW);
        }
    }

    pub(super) fn python_env_root(python_cmd: &str) -> Option<PathBuf> {
        let path = PathBuf::from(python_cmd);
        let parent = path.parent()?.to_path_buf();
        // A bare command name ("python") is found via PATH and has no env root.
        if parent.as_os_str().is_empty() {
            return None;
        }
        if let Some(dir_name) = parent.file_name().and_then(|s| s.to_str()) {
            if dir_name.eq_ignore_ascii_case("scripts") || dir_name.eq_ignore_ascii_case("bin") {
                return parent.parent().map(|p| p.to_path_buf());
            }
        }
        Some(parent)
    }

    pub(super) fn venv_python_path(venv_dir: &Path) -> PathBuf {
        if cfg!(windows) {
            venv_dir.join("Scripts").join("python.exe")
        } else {
            venv_dir.join("bin").join("python")
        }
    }
}

/// Where python.org installers put Python on Windows, newest version first:
/// per-user (`%LOCALAPPDATA%\Programs\Python\Python3XY`), then all-users
/// (`%ProgramFiles%\Python3XY`).
pub(super) fn windows_python_locations(
    local_appdata: Option<&Path>,
    program_files: Option<&Path>,
) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let roots = [
        local_appdata.map(|d| d.join("Programs").join("Python")),
        program_files.map(Path::to_path_buf),
    ];
    for root in roots.into_iter().flatten() {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        let mut found: Vec<(u32, PathBuf)> = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().to_string();
                // "Python312" -> 12; skip 32-bit "-32" and other folders.
                let minor = name.strip_prefix("Python3")?.parse::<u32>().ok()?;
                let exe = entry.path().join("python.exe");
                exe.is_file().then_some((minor, exe))
            })
            .collect();
        found.sort_by(|a, b| b.0.cmp(&a.0));
        out.extend(found.into_iter().map(|(_, exe)| exe));
    }
    out
}

/// `ffmpeg.exe` inside a winget package folder (Gyan.FFmpeg*), if installed.
fn find_winget_ffmpeg(packages: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(packages).ok()?;
    for pkg in entries.flatten() {
        let name = pkg.file_name().to_string_lossy().to_ascii_lowercase();
        if !name.starts_with("gyan.ffmpeg") {
            continue;
        }
        // Layout: <package>/ffmpeg-<version>-<build>/bin/ffmpeg.exe
        for build in std::fs::read_dir(pkg.path()).ok()?.flatten() {
            let exe = build.path().join("bin").join("ffmpeg.exe");
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    None
}

/// Whether an interpreter path is the Microsoft Store Python (or its alias).
fn is_store_python(exe: &str) -> bool {
    let lower = exe.to_ascii_lowercase().replace('/', "\\");
    lower.contains("\\windowsapps\\")
}

/// Replace characters Windows does not allow in file names (and control
/// characters) with `_`, and avoid reserved device names such as `CON` or
/// `NUL`. Applied on every platform so files can be copied to Windows.
pub(super) fn sanitize_file_name(name: &str, fallback: &str) -> String {
    let mut out: String = name
        .trim()
        .chars()
        .map(|c| match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    // Windows silently drops trailing dots and spaces.
    let trimmed_len = out.trim_end_matches(['.', ' ']).len();
    out.truncate(trimmed_len);
    if out.is_empty() {
        return fallback.to_string();
    }
    let stem = out.split('.').next().unwrap_or("").to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.as_bytes()[3].is_ascii_digit());
    if reserved {
        out.insert(0, '_');
    }
    out
}

/// Drop the `\\?\` prefix `canonicalize` adds on Windows, for display.
pub(super) fn display_path(path: &Path) -> String {
    let s = path.display().to_string();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => rest.to_string(),
        _ => s,
    }
}

/// The converter's dependencies (anndata, numpy) no longer support older Pythons.
const MIN_PYTHON_VERSION: (u32, u32) = (3, 9);

#[cfg(test)]
mod tests {
    use super::{display_path, is_store_python, sanitize_file_name, windows_python_locations};
    use std::path::Path;

    #[test]
    fn python_install_folders_are_found_newest_first() {
        let base = std::env::temp_dir().join(format!("stviz_pyloc_{}", std::process::id()));
        let local = base.join("local");
        let pf = base.join("pf");
        let user = local.join("Programs").join("Python");
        for dir in [
            user.join("Python39"),
            user.join("Python312"),
            user.join("Python311-32"),
            user.join("Launcher"),
            pf.join("Python310"),
            pf.join("Python313"),
        ] {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("python.exe"), b"").unwrap();
        }
        // A folder without python.exe is ignored.
        std::fs::create_dir_all(user.join("Python314")).unwrap();

        let found = windows_python_locations(Some(&local), Some(&pf));
        let expected = vec![
            user.join("Python312").join("python.exe"),
            user.join("Python39").join("python.exe"),
            pf.join("Python313").join("python.exe"),
            pf.join("Python310").join("python.exe"),
        ];
        assert_eq!(found, expected);
        assert!(windows_python_locations(None, None).is_empty());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn file_names_are_made_safe_for_windows() {
        assert_eq!(sanitize_file_name("my video.mp4", "x"), "my video.mp4");
        assert_eq!(sanitize_file_name("a:b*c?.mp4", "x"), "a_b_c_.mp4");
        assert_eq!(sanitize_file_name("final.  ", "x"), "final");
        assert_eq!(sanitize_file_name("   ", "x"), "x");
        assert_eq!(sanitize_file_name("con.mp4", "x"), "_con.mp4");
        assert_eq!(sanitize_file_name("COM3", "x"), "_COM3");
        assert_eq!(sanitize_file_name("console.mp4", "x"), "console.mp4");
    }

    #[test]
    fn store_python_and_verbatim_paths_are_recognised() {
        assert!(is_store_python(
            r"C:\Users\a\AppData\Local\Microsoft\WindowsApps\python.exe"
        ));
        assert!(!is_store_python(r"C:\Python312\python.exe"));
        assert_eq!(
            display_path(Path::new(r"\\?\C:\data\a.h5ad")),
            r"C:\data\a.h5ad"
        );
        assert_eq!(display_path(Path::new("/data/a.h5ad")), "/data/a.h5ad");
    }
}
