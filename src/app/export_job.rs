//! Slow export work that runs on a worker thread so the UI keeps drawing:
//! encoding the finished frames into a video (ffmpeg or the OpenCV fallback)
//! and the opt-in ffmpeg install.
//!
//! The worker reports progress as log lines over a channel and returns a
//! [`JobOutcome`]; [`StvizApp::poll_export_job`] picks both up each frame.

use super::{DirectVideoExport, StvizApp};
use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

/// How often a worker checks a running subprocess for exit or cancellation.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExportJobKind {
    Encode,
    InstallFfmpeg,
}

/// What a finished job hands back to the UI thread.
#[derive(Debug, Default)]
pub(super) struct JobOutcome {
    /// New text for the export status line.
    pub(super) status: Option<String>,
    /// Python interpreter used by the OpenCV fallback, if it ran.
    pub(super) python_exe: Option<String>,
    /// An install ran, so the cached ffmpeg path should be re-resolved.
    pub(super) ffmpeg_changed: bool,
}

/// Sends log lines from a worker to the UI thread.
pub(super) struct JobLog(mpsc::Sender<String>);

impl JobLog {
    fn line(&self, line: impl Into<String>) {
        let _ = self.0.send(line.into());
    }

    /// Log a subprocess stream under a heading, skipping it when empty.
    fn stream(&self, heading: &str, bytes: &[u8]) {
        let text = String::from_utf8_lossy(bytes);
        let text = text.trim();
        if !text.is_empty() {
            self.line(heading);
            self.line(text);
        }
    }
}

/// A running background job, owned by the app while it runs.
pub(super) struct ExportJob {
    pub(super) kind: ExportJobKind,
    rx: mpsc::Receiver<String>,
    cancel: Arc<AtomicBool>,
    handle: JoinHandle<JobOutcome>,
}

impl ExportJob {
    pub(super) fn spawn<F>(kind: ExportJobKind, work: F) -> Self
    where
        F: FnOnce(&JobLog, &AtomicBool) -> JobOutcome + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let handle = std::thread::spawn(move || work(&JobLog(tx), &worker_cancel));
        Self {
            kind,
            rx,
            cancel,
            handle,
        }
    }

    pub(super) fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub(super) fn is_cancelling(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub(super) fn drain_log(&self) -> Vec<String> {
        self.rx.try_iter().collect()
    }

    pub(super) fn is_finished(&self) -> bool {
        self.handle.is_finished()
    }

    /// Wait for the worker and return its outcome plus any log lines it sent
    /// after the last drain.
    pub(super) fn join(self) -> (JobOutcome, Vec<String>) {
        let outcome = self.handle.join().unwrap_or_else(|_| JobOutcome {
            status: Some("Export failed: background worker crashed.".to_string()),
            ..JobOutcome::default()
        });
        let lines = self.rx.try_iter().collect();
        (outcome, lines)
    }
}

/// Everything the encode worker needs, captured when the last frame is done.
pub(super) struct EncodePlan {
    pub(super) direct: Option<DirectVideoExport>,
    pub(super) frames: u32,
    pub(super) frames_dir: PathBuf,
    pub(super) temp_out: PathBuf,
    pub(super) final_out: PathBuf,
    pub(super) export_root: PathBuf,
    pub(super) output_root: PathBuf,
    pub(super) keep_frames: bool,
    pub(super) run_ffmpeg: bool,
    pub(super) ffmpeg: Option<PathBuf>,
    pub(super) install_ffmpeg: bool,
    pub(super) fps: u32,
    pub(super) crf: u32,
    pub(super) preset: &'static str,
    pub(super) project_dir: PathBuf,
    pub(super) venv_dir: PathBuf,
}

pub(super) enum RunError {
    Cancelled,
    Failed(String),
}

impl From<String> for RunError {
    fn from(msg: String) -> Self {
        Self::Failed(msg)
    }
}

/// Encode the finished frames (or wait for the streaming ffmpeg) and move the
/// result into place. Cleans up temp output and frames on failure or cancel.
pub(super) fn run_encode(mut plan: EncodePlan, log: &JobLog, cancel: &AtomicBool) -> JobOutcome {
    let mut outcome = JobOutcome::default();
    let result = encode_inner(&mut plan, log, cancel, &mut outcome);
    let status = match result {
        Ok(status) => status,
        Err(RunError::Cancelled) => {
            log.line("Export cancelled.");
            remove_frames_dir(&plan.export_root, &plan.frames_dir, &mut |s| log.line(s));
            remove_temp_output(&plan.export_root, &plan.temp_out, &mut |s| log.line(s));
            "Export cancelled.".to_string()
        }
        Err(RunError::Failed(status)) => {
            remove_temp_output(&plan.export_root, &plan.temp_out, &mut |s| log.line(s));
            status
        }
    };
    outcome.status = Some(status);
    outcome
}

fn encode_inner(
    plan: &mut EncodePlan,
    log: &JobLog,
    cancel: &AtomicBool,
    outcome: &mut JobOutcome,
) -> Result<String, RunError> {
    if let Some(mut direct) = plan.direct.take() {
        // Closing stdin tells ffmpeg the stream is over; it then flushes the
        // encoder, which can take a while on the slower presets.
        direct.stdin.take();
        log.line("Waiting for ffmpeg to finish encoding...");
        let status = wait_cancellable(&mut direct.child, cancel);
        let stderr = std::fs::read_to_string(&direct.stderr_path).unwrap_or_default();
        if !stderr.trim().is_empty() {
            log.line("ffmpeg output:");
            log.line(stderr.trim());
        }
        let status = status?;
        if !status.success() {
            log.line(format!("Direct ffmpeg export failed: {status}"));
            return Err(format!("ffmpeg failed: {status}").into());
        }
        finalize(plan, log)?;
        log.line("Direct video export succeeded.");
        remove_frames_dir(&plan.export_root, &plan.frames_dir, &mut |s| log.line(s));
        return Ok(format!("Wrote video: {}", plan.final_out.display()));
    }

    if !plan.run_ffmpeg {
        log.line("Export finished (ffmpeg disabled).");
        if !plan.keep_frames {
            remove_frames_dir(&plan.export_root, &plan.frames_dir, &mut |s| log.line(s));
        }
        return Ok(format!(
            "Exported {} frames to {}",
            plan.frames,
            plan.frames_dir.display()
        ));
    }

    if plan.ffmpeg.is_none() && plan.install_ffmpeg {
        outcome.ffmpeg_changed = true;
        plan.ffmpeg = install_ffmpeg(&plan.project_dir, log, cancel)?;
    }

    let (label, kept_suffix) = if let Some(ffmpeg) = plan.ffmpeg.clone() {
        encode_png_frames_ffmpeg(plan, &ffmpeg, log, cancel)?;
        ("", " (kept frames)")
    } else {
        log.line("ffmpeg not found; trying OpenCV fallback.");
        match run_python_cv2_export(plan, log, cancel) {
            Ok(python_exe) => outcome.python_exe = Some(python_exe),
            Err(RunError::Failed(e)) => {
                log.line(format!("OpenCV fallback failed: {e}"));
                return Err(format!(
                    "Exported {} frames to {} (OpenCV fallback failed)",
                    plan.frames,
                    plan.frames_dir.display()
                )
                .into());
            }
            Err(RunError::Cancelled) => return Err(RunError::Cancelled),
        }
        (" (OpenCV)", " (kept frames)")
    };

    finalize(plan, log)?;
    let mut status = format!("Wrote video{label}: {}", plan.final_out.display());
    if plan.keep_frames {
        status.push_str(kept_suffix);
        log.line("Video render succeeded (frames kept).");
    } else {
        log.line("Video render succeeded; cleaning PNG frames.");
        remove_frames_dir(&plan.export_root, &plan.frames_dir, &mut |s| log.line(s));
    }
    Ok(status)
}

fn finalize(plan: &EncodePlan, log: &JobLog) -> Result<(), RunError> {
    finalize_output(
        &plan.export_root,
        &plan.output_root,
        &plan.temp_out,
        &plan.final_out,
    )
    .map_err(|e| {
        log.line(format!("Export finalize failed: {e}"));
        RunError::Failed(format!("Export finalize failed: {e}"))
    })?;
    log.line(format!("Finalized output: {}", plan.final_out.display()));
    Ok(())
}

fn encode_png_frames_ffmpeg(
    plan: &EncodePlan,
    ffmpeg: &Path,
    log: &JobLog,
    cancel: &AtomicBool,
) -> Result<(), RunError> {
    let pattern = plan.frames_dir.join("frame_%06d.png");
    let mut cmd = Command::new(ffmpeg);
    StvizApp::apply_subprocess_flags(&mut cmd);
    cmd.arg("-y")
        .arg("-framerate")
        .arg(plan.fps.to_string())
        .arg("-i")
        .arg(&pattern)
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg(plan.preset)
        .arg("-crf")
        .arg(plan.crf.to_string())
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&plan.temp_out);
    log.line("Encoding PNG frames with ffmpeg...");
    let out = run_cancellable(&mut cmd, cancel).map_err(|e| match e {
        RunError::Failed(e) => {
            log.line(format!("ffmpeg error: {e}"));
            RunError::Failed(format!("ffmpeg error: {e}"))
        }
        other => other,
    })?;
    if !out.status.success() {
        log.line(format!("ffmpeg failed: {}", out.status));
        log.stream("ffmpeg stdout:", &out.stdout);
        log.stream("ffmpeg stderr:", &out.stderr);
        return Err(format!("ffmpeg failed: {}", out.status).into());
    }
    Ok(())
}

/// Encode with `python/export_video_cv2.py` in the managed venv, creating the
/// venv and installing OpenCV on first use. Returns the interpreter used.
fn run_python_cv2_export(
    plan: &EncodePlan,
    log: &JobLog,
    cancel: &AtomicBool,
) -> Result<String, RunError> {
    let script = plan.project_dir.join("python").join("export_video_cv2.py");
    if !script.exists() {
        return Err(format!("OpenCV exporter not found: {}", script.display()).into());
    }
    let (python_cmd, python_exe) = StvizApp::detect_python_cmd()?;
    let venv_dir = &plan.venv_dir;
    let venv_python = StvizApp::venv_python_path(venv_dir);
    let base_env_root = StvizApp::python_env_root(&python_cmd);

    // Run one setup step, logging its output if it fails.
    let step = |cmd: &mut Command, what: &str| -> Result<(), RunError> {
        let out = run_cancellable(cmd, cancel)
            .map_err(|e| prefix_failure(e, &format!("Failed to {what}")))?;
        if !out.status.success() {
            log.stream("stdout:", &out.stdout);
            log.stream("stderr:", &out.stderr);
            return Err(format!("Failed to {what} ({}).", out.status).into());
        }
        Ok(())
    };

    if !venv_python.exists() {
        log.line("Creating managed converter environment for OpenCV fallback...");
        let mut cmd = Command::new(&python_cmd);
        cmd.arg("-m").arg("venv").arg(venv_dir);
        StvizApp::apply_python_env(&mut cmd, base_env_root.as_deref());
        step(&mut cmd, "create virtual environment")?;
    }
    let python_exe = if python_exe.is_empty() {
        venv_python.display().to_string()
    } else {
        python_exe
    };

    log.line("Checking OpenCV (cv2) dependency...");
    let mut check = Command::new(&venv_python);
    check.arg("-c").arg("import cv2");
    StvizApp::apply_python_env(&mut check, Some(venv_dir));
    let has_cv2 = run_cancellable(&mut check, cancel)
        .map_err(|e| prefix_failure(e, "Failed to check OpenCV dependency"))?
        .status
        .success();

    if !has_cv2 {
        log.line("Updating Python packaging tools...");
        let mut cmd = Command::new(&venv_python);
        cmd.args([
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]);
        StvizApp::apply_python_env(&mut cmd, Some(venv_dir));
        step(&mut cmd, "bootstrap pip tooling")?;

        log.line("Installing OpenCV dependency...");
        let mut cmd = Command::new(&venv_python);
        cmd.args([
            "-m",
            "pip",
            "install",
            "--prefer-binary",
            "opencv-python-headless",
        ]);
        StvizApp::apply_python_env(&mut cmd, Some(venv_dir));
        step(&mut cmd, "install OpenCV dependency")?;
    }

    log.line("Encoding video with OpenCV...");
    let mut cmd = Command::new(&venv_python);
    cmd.arg("-X")
        .arg("faulthandler")
        .arg(&script)
        .arg("--input-dir")
        .arg(&plan.frames_dir)
        .arg("--output")
        .arg(&plan.temp_out)
        .arg("--fps")
        .arg(plan.fps.to_string())
        .env("PYTHONFAULTHANDLER", "1")
        .current_dir(&plan.project_dir);
    StvizApp::apply_python_env(&mut cmd, Some(venv_dir));
    let out = run_cancellable(&mut cmd, cancel)
        .map_err(|e| prefix_failure(e, "Failed to run OpenCV exporter"))?;
    if !out.status.success() {
        log.stream("stdout:", &out.stdout);
        log.stream("stderr:", &out.stderr);
        return Err(format!("OpenCV export failed ({}).", out.status).into());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    if !stdout.trim().is_empty() {
        log.line(stdout.trim());
    }
    Ok(python_exe)
}

fn prefix_failure(e: RunError, prefix: &str) -> RunError {
    match e {
        RunError::Failed(msg) => RunError::Failed(format!("{prefix}: {msg}")),
        other => other,
    }
}

/// Try the platform's package managers in turn until ffmpeg can be found.
/// Only runs when the user turned on auto-install.
pub(super) fn install_ffmpeg(
    project_dir: &Path,
    log: &JobLog,
    cancel: &AtomicBool,
) -> Result<Option<PathBuf>, RunError> {
    log.line("ffmpeg not found; trying to install it (internet required).");
    for (label, program, args) in ffmpeg_installers() {
        log.line(format!(
            "Trying ffmpeg installer ({label}): {program} {}",
            args.join(" ")
        ));
        let mut cmd = Command::new(program);
        StvizApp::apply_subprocess_flags(&mut cmd);
        cmd.args(args.iter());
        match run_cancellable(&mut cmd, cancel) {
            Ok(out) => {
                log.stream("stdout:", &out.stdout);
                log.stream("stderr:", &out.stderr);
                if !out.status.success() {
                    log.line(format!("Installer failed ({label}): {}", out.status));
                    continue;
                }
                if let Some(path) = StvizApp::resolve_ffmpeg_path(project_dir) {
                    log.line(format!("ffmpeg detected after {label} install."));
                    return Ok(Some(path));
                }
            }
            Err(RunError::Cancelled) => return Err(RunError::Cancelled),
            Err(RunError::Failed(e)) => {
                log.line(format!("Installer unavailable ({label}): {e}"));
            }
        }
    }
    let path = StvizApp::resolve_ffmpeg_path(project_dir);
    if path.is_none() {
        log.line("Automatic ffmpeg install did not complete.");
    }
    Ok(path)
}

type Installer = (&'static str, &'static str, &'static [&'static str]);

fn ffmpeg_installers() -> &'static [Installer] {
    #[cfg(target_os = "windows")]
    {
        &[
            (
                "winget",
                "winget",
                &[
                    "install",
                    "-e",
                    "--id",
                    "Gyan.FFmpeg",
                    "--silent",
                    "--scope",
                    "user",
                    "--accept-source-agreements",
                    "--accept-package-agreements",
                    "--disable-interactivity",
                ],
            ),
            ("choco", "choco", &["install", "ffmpeg", "-y"]),
        ]
    }
    #[cfg(target_os = "macos")]
    {
        &[("homebrew", "brew", &["install", "ffmpeg"])]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        &[
            (
                "apt (sudo)",
                "sudo",
                &["-n", "apt-get", "install", "-y", "ffmpeg"],
            ),
            ("apt", "apt-get", &["install", "-y", "ffmpeg"]),
            ("dnf", "dnf", &["install", "-y", "ffmpeg"]),
            ("pacman", "pacman", &["-S", "--noconfirm", "ffmpeg"]),
            (
                "zypper",
                "zypper",
                &["--non-interactive", "install", "ffmpeg"],
            ),
        ]
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        &[]
    }
}

struct ProcessOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Like `Command::output`, but kills the process if `cancel` is set.
fn run_cancellable(cmd: &mut Command, cancel: &AtomicBool) -> Result<ProcessOutput, RunError> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| RunError::Failed(e.to_string()))?;
    // Drain both pipes on their own threads so a chatty process can't fill a
    // pipe buffer and stall while we wait for it.
    let stdout = child.stdout.take().map(read_all_in_background);
    let stderr = child.stderr.take().map(read_all_in_background);
    let status = wait_cancellable(&mut child, cancel);
    let collect =
        |h: Option<JoinHandle<Vec<u8>>>| h.and_then(|h| h.join().ok()).unwrap_or_default();
    let stdout = collect(stdout);
    let stderr = collect(stderr);
    Ok(ProcessOutput {
        status: status?,
        stdout,
        stderr,
    })
}

fn read_all_in_background<R: Read + Send + 'static>(mut pipe: R) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = pipe.read_to_end(&mut buf);
        buf
    })
}

fn wait_cancellable(child: &mut Child, cancel: &AtomicBool) -> Result<ExitStatus, RunError> {
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(RunError::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => std::thread::sleep(POLL_INTERVAL),
            Err(e) => return Err(RunError::Failed(format!("wait failed: {e}"))),
        }
    }
}

/// Move the temp video to its final name, refusing paths outside the managed
/// export and output folders.
pub(super) fn finalize_output(
    export_root: &Path,
    output_root: &Path,
    temp_path: &Path,
    final_path: &Path,
) -> Result<(), String> {
    if !StvizApp::path_within_root(export_root, temp_path) {
        return Err(format!(
            "Refusing to finalize output outside managed export root: {}",
            temp_path.display()
        ));
    }
    if !StvizApp::path_within_root(output_root, final_path) {
        return Err(format!(
            "Refusing to write output outside managed output root: {}",
            final_path.display()
        ));
    }
    if !temp_path.exists() {
        return Err(format!(
            "Temporary export output missing: {}",
            temp_path.display()
        ));
    }
    if let Some(parent) = final_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if final_path.exists() {
        std::fs::remove_file(final_path)
            .map_err(|e| format!("Failed to replace existing output file: {e}"))?;
    }
    std::fs::rename(temp_path, final_path).map_err(|e| format!("Failed to finalize output: {e}"))
}

/// Delete a `loop_*` frames folder, but only inside the managed export root.
pub(super) fn remove_frames_dir(export_root: &Path, frames_dir: &Path, log: &mut dyn FnMut(&str)) {
    let is_loop_dir = frames_dir
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.starts_with("loop_"));
    if !is_loop_dir || !StvizApp::path_within_root(export_root, frames_dir) {
        log(&format!(
            "Refused to remove unmanaged frames directory: {}",
            frames_dir.display()
        ));
        return;
    }
    if !frames_dir.exists() {
        return;
    }
    match std::fs::remove_dir_all(frames_dir) {
        Ok(()) => log(&format!(
            "Removed frames directory: {}",
            frames_dir.display()
        )),
        Err(e) => log(&format!(
            "Failed to remove frames: {e} ({})",
            frames_dir.display()
        )),
    }
}

/// Delete a temporary video file, but only inside the managed export root.
pub(super) fn remove_temp_output(export_root: &Path, path: &Path, log: &mut dyn FnMut(&str)) {
    if !StvizApp::path_within_root(export_root, path) {
        log(&format!(
            "Refused to remove unmanaged temp output: {}",
            path.display()
        ));
        return;
    }
    if !path.exists() {
        return;
    }
    match std::fs::remove_file(path) {
        Ok(()) => log(&format!(
            "Removed temporary output file: {}",
            path.display()
        )),
        Err(e) => log(&format!(
            "Failed to remove temporary output: {e} ({})",
            path.display()
        )),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn run_cancellable_captures_output() {
        let cancel = AtomicBool::new(false);
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "echo out; echo err >&2; exit 3"]);
        let Ok(out) = run_cancellable(&mut cmd, &cancel) else {
            panic!("sh should run");
        };
        assert_eq!(out.status.code(), Some(3));
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "out");
        assert_eq!(String::from_utf8_lossy(&out.stderr).trim(), "err");
    }

    #[test]
    fn run_cancellable_kills_on_cancel() {
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let setter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            flag.store(true, Ordering::Relaxed);
        });
        let start = std::time::Instant::now();
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        let result = run_cancellable(&mut cmd, &cancel);
        setter.join().unwrap();
        assert!(matches!(result, Err(RunError::Cancelled)));
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn job_reports_log_lines_and_outcome() {
        let job = ExportJob::spawn(ExportJobKind::Encode, |log, _cancel| {
            log.line("hello");
            JobOutcome {
                status: Some("done".to_string()),
                ..JobOutcome::default()
            }
        });
        let (outcome, lines) = job.join();
        assert_eq!(outcome.status.as_deref(), Some("done"));
        assert_eq!(lines, vec!["hello".to_string()]);
    }

    #[test]
    fn finalize_refuses_paths_outside_roots() {
        let base = std::env::temp_dir().join(format!("stviz_finalize_{}", std::process::id()));
        let export_root = base.join("exports");
        let output_root = base.join("output");
        std::fs::create_dir_all(&export_root).unwrap();
        std::fs::create_dir_all(&output_root).unwrap();
        let temp = export_root.join("video.tmp.mp4");
        std::fs::write(&temp, b"x").unwrap();

        let outside = base.join("elsewhere.mp4");
        assert!(finalize_output(&export_root, &output_root, &temp, &outside).is_err());

        let dest = output_root.join("video.mp4");
        finalize_output(&export_root, &output_root, &temp, &dest).unwrap();
        assert!(dest.exists() && !temp.exists());
        let _ = std::fs::remove_dir_all(&base);
    }
}

#[cfg(all(test, unix))]
mod encode_tests {
    use super::*;

    /// Encodes real PNG frames with ffmpeg off the UI thread, end to end.
    /// Skipped when ffmpeg isn't installed.
    #[test]
    fn encodes_png_frames_and_cleans_up() {
        let Some(ffmpeg) = StvizApp::resolve_ffmpeg_path(Path::new(".")) else {
            eprintln!("ffmpeg not installed; skipping");
            return;
        };
        let base = std::env::temp_dir().join(format!("stviz_encode_{}", std::process::id()));
        let export_root = base.join("exports");
        let output_root = base.join("output");
        let frames_dir = export_root.join("loop_test");
        std::fs::create_dir_all(&frames_dir).unwrap();
        std::fs::create_dir_all(&output_root).unwrap();
        for i in 0..4u8 {
            let img = image::RgbaImage::from_pixel(32, 32, image::Rgba([i * 60, 0, 0, 255]));
            img.save(frames_dir.join(format!("frame_{i:06}.png")))
                .unwrap();
        }
        let plan = EncodePlan {
            direct: None,
            frames: 4,
            frames_dir: frames_dir.clone(),
            temp_out: export_root.join("out.tmp.mp4"),
            final_out: output_root.join("out.mp4"),
            export_root: export_root.clone(),
            output_root: output_root.clone(),
            keep_frames: false,
            run_ffmpeg: true,
            ffmpeg: Some(ffmpeg),
            install_ffmpeg: false,
            fps: 4,
            crf: 23,
            preset: "fast",
            project_dir: PathBuf::from("."),
            venv_dir: base.join("venv"),
        };
        let job = ExportJob::spawn(ExportJobKind::Encode, move |log, cancel| {
            run_encode(plan, log, cancel)
        });
        let (outcome, lines) = job.join();
        let status = outcome.status.unwrap_or_default();
        assert!(status.starts_with("Wrote video"), "{status}\n{lines:?}");
        assert!(output_root.join("out.mp4").metadata().unwrap().len() > 0);
        assert!(!frames_dir.exists(), "frames should be removed");
        let _ = std::fs::remove_dir_all(&base);
    }
}
