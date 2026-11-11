//! .h5ad conversion and mock dataset generation via the bundled Python scripts.
//!
//! The work runs on a worker thread ([`ConvertJob`]). It streams the scripts'
//! output back as log lines, can be cancelled, and sets up the managed Python
//! environment on first use.

use super::{
    process::{run_cancellable, run_streaming, RunError},
    util::chrono_like_timestamp,
    StvizApp,
};
use crate::data::Dataset;
use rand::Rng;
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread::JoinHandle,
};

/// Which matrix of the .h5ad supplies gene expression.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum ExprSource {
    #[default]
    X,
    Raw,
    Layer,
}

pub(super) struct ConvertResult {
    msg: String,
    output: PathBuf,
    load_after: bool,
    python_exe: Option<String>,
}

/// A running conversion, owned by the app while it runs.
pub(super) struct ConvertJob {
    rx: mpsc::Receiver<String>,
    cancel: Arc<AtomicBool>,
    handle: JoinHandle<Result<ConvertResult, String>>,
}

impl ConvertJob {
    fn spawn(req: ConvertRequest) -> Self {
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let handle = std::thread::spawn(move || {
            let mut log_file = req.log_path.as_ref().and_then(|p| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .ok()
            });
            let mut log = |line: &str| {
                if let Some(file) = log_file.as_mut() {
                    let _ = writeln!(file, "{line}");
                }
                let _ = tx.send(line.to_string());
            };
            let result = run_convert(&req, &mut log, &worker_cancel);
            if result.is_err() {
                req.remove_partial_outputs();
            }
            result.map_err(|e| {
                let msg = match e {
                    RunError::Cancelled => "Conversion cancelled.".to_string(),
                    RunError::Failed(msg) => msg,
                };
                log(&msg);
                match req.log_path.as_ref() {
                    Some(path) if !msg.ends_with("cancelled.") => {
                        format!("{msg}\nLog: {}", path.display())
                    }
                    _ => msg,
                }
            })
        });
        Self { rx, cancel, handle }
    }

    pub(super) fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub(super) fn is_cancelling(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Stop the job and wait for its subprocess to be killed.
    pub(super) fn abort(self) {
        self.cancel();
        let _ = self.handle.join();
    }
}

struct MockRequest {
    cells: u32,
    seed: u64,
}

struct ConvertRequest {
    mock: Option<MockRequest>,
    input: PathBuf,
    output: PathBuf,
    /// Stored in the .stviz so a later conversion can tell whose output it is.
    source: String,
    include_expr: bool,
    expr_source: ExprSource,
    layer: String,
    gene_symbols: String,
    load_after: bool,
    project_dir: PathBuf,
    venv_dir: PathBuf,
    log_path: Option<PathBuf>,
}

impl ConvertRequest {
    fn remove_partial_outputs(&self) {
        let mut partial = self.output.clone().into_os_string();
        partial.push(".partial");
        let _ = std::fs::remove_file(PathBuf::from(partial));
        if self.mock.is_some() {
            let _ = std::fs::remove_file(&self.input);
        }
    }
}

/// Run a quiet setup step, logging its output only if it fails.
fn setup_step(
    cmd: &mut Command,
    what: &str,
    log: &mut dyn FnMut(&str),
    cancel: &AtomicBool,
) -> Result<(), RunError> {
    let out = run_cancellable(cmd, cancel).map_err(|e| match e {
        RunError::Failed(msg) => RunError::Failed(format!("Failed to {what}: {msg}")),
        other => other,
    })?;
    if !out.status.success() {
        for stream in [&out.stdout, &out.stderr] {
            let text = String::from_utf8_lossy(stream);
            if !text.trim().is_empty() {
                log(text.trim());
            }
        }
        return Err(RunError::Failed(format!(
            "Failed to {what} ({}).",
            out.status
        )));
    }
    Ok(())
}

/// Run a script, streaming its output into the log. On failure the script's
/// own `Error: ...` line (if any) becomes the error message.
fn run_script(
    cmd: &mut Command,
    what: &str,
    log: &mut dyn FnMut(&str),
    cancel: &AtomicBool,
) -> Result<(), RunError> {
    let mut error_line = None;
    let status = run_streaming(cmd, cancel, &mut |line| {
        if let Some(rest) = line.strip_prefix("Error: ") {
            error_line = Some(rest.to_string());
        }
        log(line);
    })
    .map_err(|e| match e {
        RunError::Failed(msg) => RunError::Failed(format!("Failed to run {what}: {msg}")),
        other => other,
    })?;
    if status.success() {
        return Ok(());
    }
    Err(RunError::Failed(match error_line {
        Some(reason) => format!("{what} failed: {reason}"),
        None => format!("{what} failed ({status})."),
    }))
}

/// Make sure the managed venv exists and has the converter dependencies.
/// Returns the venv's Python. Needs internet only when something is missing.
fn ensure_converter_env(
    project_dir: &Path,
    venv_dir: &Path,
    log: &mut dyn FnMut(&str),
    cancel: &AtomicBool,
) -> Result<PathBuf, RunError> {
    let venv_python = StvizApp::venv_python_path(venv_dir);
    if !venv_python.exists() {
        let (python_cmd, python_exe) = StvizApp::detect_python_cmd()?;
        log(&format!("Using Python: {python_exe}"));
        log("Creating the converter environment (first run only)...");
        let mut cmd = Command::new(&python_cmd);
        cmd.arg("-m").arg("venv").arg(venv_dir);
        StvizApp::apply_python_env(&mut cmd, StvizApp::python_env_root(&python_cmd).as_deref());
        setup_step(&mut cmd, "create the converter environment", log, cancel)?;
    }

    let mut check = Command::new(&venv_python);
    check
        .arg("-c")
        .arg("import anndata, h5py, numpy, pandas, scipy");
    StvizApp::apply_python_env(&mut check, Some(venv_dir));
    let has_deps = run_cancellable(&mut check, cancel)
        .map_err(|e| match e {
            RunError::Failed(msg) => {
                RunError::Failed(format!("Failed to check converter dependencies: {msg}"))
            }
            other => other,
        })?
        .status
        .success();
    if has_deps {
        return Ok(venv_python);
    }

    log("Installing converter dependencies (first run only, needs internet)...");
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
    setup_step(&mut cmd, "update pip", log, cancel)?;

    let requirements = project_dir.join("python").join("requirements.txt");
    let mut cmd = Command::new(&venv_python);
    cmd.args([
        "-m",
        "pip",
        "install",
        "--prefer-binary",
        "--progress-bar",
        "off",
    ]);
    if requirements.exists() {
        cmd.arg("-r").arg(&requirements);
    } else {
        cmd.args(["anndata", "h5py", "numpy", "pandas", "scipy"]);
    }
    StvizApp::apply_python_env(&mut cmd, Some(venv_dir));
    run_script(&mut cmd, "Dependency install", log, cancel)?;
    log("Dependencies installed.");
    Ok(venv_python)
}

fn python_script_command(venv_python: &Path, venv_dir: &Path, project_dir: &Path) -> Command {
    let mut cmd = Command::new(venv_python);
    cmd.arg("-X")
        .arg("faulthandler")
        .env("PYTHONFAULTHANDLER", "1")
        .env("PYTHONUNBUFFERED", "1")
        .current_dir(project_dir);
    StvizApp::apply_python_env(&mut cmd, Some(venv_dir));
    cmd
}

fn run_convert(
    req: &ConvertRequest,
    log: &mut dyn FnMut(&str),
    cancel: &AtomicBool,
) -> Result<ConvertResult, RunError> {
    let scripts = req.project_dir.join("python");
    let exporter = scripts.join("export_stviz.py");
    if !exporter.exists() {
        return Err(RunError::Failed(format!(
            "Exporter not found: {}",
            exporter.display()
        )));
    }
    if req.mock.is_none() {
        log(&format!("Input file: {}", req.input.display()));
        if !req.input.is_file() {
            return Err(RunError::Failed(format!(
                "Input file not found: {}",
                req.input.display()
            )));
        }
    }
    log(&format!("Output file: {}", req.output.display()));

    let venv_python = ensure_converter_env(&req.project_dir, &req.venv_dir, log, cancel)?;

    if let Some(mock) = req.mock.as_ref() {
        let generator = scripts.join("mock_spatial_dataset.py");
        if !generator.exists() {
            return Err(RunError::Failed(format!(
                "Mock dataset script not found: {}",
                generator.display()
            )));
        }
        log(&format!(
            "Generating a mock dataset with {} cells...",
            mock.cells
        ));
        let mut cmd = python_script_command(&venv_python, &req.venv_dir, &req.project_dir);
        cmd.arg(&generator)
            .arg("--out")
            .arg(&req.input)
            .arg("--cells")
            .arg(mock.cells.to_string())
            .arg("--seed")
            .arg(mock.seed.to_string());
        if !req.include_expr {
            cmd.arg("--no-expr");
        }
        run_script(&mut cmd, "Mock dataset generation", log, cancel)?;
    }

    log(if req.include_expr {
        "Converting (with gene expression)..."
    } else {
        "Converting (without gene expression)..."
    });
    let mut cmd = python_script_command(&venv_python, &req.venv_dir, &req.project_dir);
    cmd.arg(&exporter)
        .arg("--input")
        .arg(&req.input)
        .arg("--output")
        .arg(&req.output)
        .arg("--source")
        .arg(&req.source);
    if req.include_expr {
        cmd.arg("--include-expr");
        match req.expr_source {
            ExprSource::X => {}
            ExprSource::Raw => {
                cmd.arg("--use-raw");
            }
            ExprSource::Layer => {
                cmd.arg("--layer").arg(req.layer.trim());
            }
        }
        if !req.gene_symbols.trim().is_empty() {
            cmd.arg("--gene-symbols").arg(req.gene_symbols.trim());
        }
    }
    run_script(&mut cmd, "Conversion", log, cancel)?;

    Ok(ConvertResult {
        msg: if req.mock.is_some() {
            "Mock dataset ready.".to_string()
        } else {
            "Conversion completed.".to_string()
        },
        output: req.output.clone(),
        load_after: req.load_after,
        python_exe: Some(venv_python.display().to_string()),
    })
}

impl StvizApp {
    pub(super) fn convert_running(&self) -> bool {
        self.convert_job.is_some()
    }

    pub(super) fn poll_convert_job(&mut self) {
        let Some(job) = self.convert_job.as_ref() else {
            return;
        };
        let cancelling = job.is_cancelling();
        let finished = job.handle.is_finished();
        let lines: Vec<String> = job.rx.try_iter().collect();
        for line in lines {
            if !cancelling {
                self.convert_status = Some(line.clone());
            }
            self.append_convert_log(&line);
        }
        if !finished {
            return;
        }
        let job = self.convert_job.take().unwrap();
        let lines: Vec<String> = job.rx.try_iter().collect();
        for line in lines {
            self.append_convert_log(&line);
        }
        match job.handle.join() {
            Ok(Ok(result)) => {
                if let Some(python_exe) = result.python_exe {
                    self.convert_last_python_exe = Some(python_exe);
                }
                self.convert_status = Some(result.msg);
                if result.load_after {
                    self.start_dataset_load(&result.output);
                }
            }
            Ok(Err(msg)) => self.convert_status = Some(msg),
            Err(_) => self.convert_status = Some("Conversion failed unexpectedly.".to_string()),
        }
    }

    fn append_convert_log(&mut self, line: &str) {
        if !self.convert_log_text.is_empty() {
            self.convert_log_text.push('\n');
        }
        self.convert_log_text.push_str(line);
    }

    pub(super) fn cancel_convert(&mut self) {
        if let Some(job) = self.convert_job.as_ref() {
            job.cancel();
            self.convert_status = Some("Cancelling...".to_string());
        }
    }

    pub(super) fn queue_h5ad_convert(&mut self, path: &Path) {
        if self.convert_running() {
            self.convert_status =
                Some("A conversion is already running; wait for it or cancel it.".to_string());
            return;
        }
        if self.convert_include_expr
            && self.convert_expr_source == ExprSource::Layer
            && self.convert_layer.trim().is_empty()
        {
            self.convert_status = Some("Enter a layer name, or pick another source.".to_string());
            return;
        }
        let source = std::fs::canonicalize(path)
            .unwrap_or_else(|_| path.to_path_buf())
            .display()
            .to_string();
        let output = self.convert_output_for(path, &source);
        self.release_dataset_file(&output);
        let log_path = self
            .logs_dir
            .join(format!("convert_log_{}.txt", chrono_like_timestamp()));
        self.start_convert_job(ConvertRequest {
            mock: None,
            input: path.to_path_buf(),
            output,
            source,
            include_expr: self.convert_include_expr,
            expr_source: self.convert_expr_source,
            layer: self.convert_layer.clone(),
            gene_symbols: self.convert_gene_symbols.clone(),
            load_after: !self.convert_generate_only,
            project_dir: self.project_dir.clone(),
            venv_dir: self.converter_venv_dir.clone(),
            log_path: Some(log_path),
        });
    }

    pub(super) fn start_mock_dataset(&mut self) {
        if self.convert_running() {
            return;
        }
        self.cleanup_previous_mock();
        let cells = self.mock_cells.clamp(10_000, 10_000_000);
        self.mock_cells = cells;
        let _ = std::fs::create_dir_all(&self.output_dir);
        let ts = chrono_like_timestamp();
        let h5ad = self
            .output_dir
            .join(format!("mock_spatial_{cells}_{ts}.h5ad"));
        let stviz = self
            .output_dir
            .join(format!("mock_spatial_{cells}_{ts}.stviz"));
        self.mock_last_h5ad = Some(h5ad.clone());
        self.mock_last_stviz = Some(stviz.clone());
        self.start_convert_job(ConvertRequest {
            mock: Some(MockRequest {
                cells,
                seed: rand::thread_rng().gen(),
            }),
            source: h5ad.display().to_string(),
            input: h5ad,
            output: stviz,
            include_expr: self.convert_include_expr,
            // The generator only writes X.
            expr_source: ExprSource::X,
            layer: String::new(),
            gene_symbols: String::new(),
            load_after: !self.convert_generate_only,
            project_dir: self.project_dir.clone(),
            venv_dir: self.converter_venv_dir.clone(),
            log_path: None,
        });
    }

    fn start_convert_job(&mut self, req: ConvertRequest) {
        self.convert_log_text.clear();
        self.convert_log_path = req.log_path.clone();
        self.convert_status = Some("Preparing the converter...".to_string());
        self.convert_job = Some(ConvertJob::spawn(req));
    }

    /// Where the converted file goes: `<output>/<stem>.stviz`, or a numbered
    /// name if that file was converted from a different .h5ad.
    fn convert_output_for(&self, input: &Path, source: &str) -> PathBuf {
        let _ = std::fs::create_dir_all(&self.output_dir);
        let stem = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("converted");
        pick_convert_output(&self.output_dir, stem, source)
    }

    /// On Windows a memory-mapped file cannot be replaced, so close the loaded
    /// dataset if the conversion is about to overwrite it.
    fn release_dataset_file(&mut self, output: &Path) {
        if !cfg!(windows) {
            return;
        }
        let same = match (self.dataset_path.as_ref(), std::fs::canonicalize(output)) {
            (Some(loaded), Ok(out)) => std::fs::canonicalize(loaded).is_ok_and(|p| p == out),
            _ => false,
        };
        if same {
            self.unload_dataset();
        }
    }

    pub(super) fn cleanup_mock_artifacts(output_dir: &Path) {
        let Ok(entries) = std::fs::read_dir(output_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.starts_with("mock_spatial_") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    fn cleanup_previous_mock(&mut self) {
        for path in [self.mock_last_h5ad.take(), self.mock_last_stviz.take()]
            .into_iter()
            .flatten()
        {
            if Self::path_within_root(&self.output_dir, &path) {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

fn pick_convert_output(output_dir: &Path, stem: &str, source: &str) -> PathBuf {
    for n in 1..1000 {
        let name = if n == 1 {
            format!("{stem}.stviz")
        } else {
            format!("{stem}-{n}.stviz")
        };
        let candidate = output_dir.join(name);
        if !candidate.exists() {
            return candidate;
        }
        // Files from before the source was recorded are treated as ours.
        match Dataset::peek_source(&candidate) {
            Some(existing) if existing != source => continue,
            _ => return candidate,
        }
    }
    output_dir.join(format!("{stem}-{}.stviz", chrono_like_timestamp()))
}

#[cfg(test)]
mod tests {
    use super::pick_convert_output;
    use crate::data::test_support::write_stviz;

    #[test]
    fn output_name_avoids_other_sources() {
        let dir = std::env::temp_dir().join(format!("stviz_pick_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            pick_convert_output(&dir, "a", "/x/a.h5ad"),
            dir.join("a.stviz")
        );

        write_stviz(&dir.join("a.stviz"), 4, Some("/x/a.h5ad"));
        // Same source: reuse the name. Different source: next free name.
        assert_eq!(
            pick_convert_output(&dir, "a", "/x/a.h5ad"),
            dir.join("a.stviz")
        );
        assert_eq!(
            pick_convert_output(&dir, "a", "/y/a.h5ad"),
            dir.join("a-2.stviz")
        );

        write_stviz(&dir.join("a-2.stviz"), 4, Some("/y/a.h5ad"));
        assert_eq!(
            pick_convert_output(&dir, "a", "/y/a.h5ad"),
            dir.join("a-2.stviz")
        );
        assert_eq!(
            pick_convert_output(&dir, "a", "/z/a.h5ad"),
            dir.join("a-3.stviz")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Full mock generation + conversion through the real worker. Creates a
    /// Python venv and installs the converter dependencies, so it needs Python
    /// and internet: run with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn converts_a_mock_dataset_end_to_end() {
        use super::{ConvertJob, ConvertRequest, ExprSource, MockRequest};
        use crate::data::Dataset;

        let dir = std::env::temp_dir().join(format!("stviz_e2e_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let output = dir.join("mock.stviz");
        let job = ConvertJob::spawn(ConvertRequest {
            mock: Some(MockRequest {
                cells: 10_000,
                seed: 7,
            }),
            input: dir.join("mock.h5ad"),
            output: output.clone(),
            source: "mock".to_string(),
            include_expr: true,
            expr_source: ExprSource::X,
            layer: String::new(),
            gene_symbols: String::new(),
            load_after: false,
            project_dir: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            venv_dir: std::env::temp_dir().join("stviz_e2e_venv"),
            log_path: Some(dir.join("log.txt")),
        });
        let result = job.handle.join().unwrap();
        let lines: Vec<String> = job.rx.try_iter().collect();
        assert!(result.is_ok(), "{:?}\n{}", result.err(), lines.join("\n"));
        assert!(lines.iter().any(|l| l.starts_with("Writing expression")));
        let ds = Dataset::load(&output).unwrap();
        assert_eq!(ds.meta.n_points, 10_000);
        assert_eq!(ds.meta.source.as_deref(), Some("mock"));
        assert!(ds.meta.expr.is_some());
        assert!(!dir.join("mock.stviz.partial").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
