//! .h5ad conversion and mock dataset generation via the bundled Python scripts.

use super::*;

impl StvizApp {
    pub(super) fn poll_convert_job(&mut self) {
        let Some(handle) = self.convert_handle.as_ref() else {
            return;
        };
        if !handle.is_finished() {
            return;
        }
        let handle = self.convert_handle.take().unwrap();
        self.convert_running = false;
        match handle.join() {
            Ok(Ok(result)) => {
                if let Some(python_exe) = result.python_exe {
                    self.convert_last_python_exe = Some(python_exe);
                }
                self.convert_status = Some(result.msg);
                if result.load_after {
                    if !result.output.exists() {
                        let msg = format!(
                            "Load skipped: output not found at {}",
                            result.output.display()
                        );
                        self.convert_status = Some(msg.clone());
                        self.last_error = Some(msg);
                        return;
                    }
                    if let Err(e) = self.load_dataset(&result.output) {
                        let msg = format!("Load failed: {e:#}");
                        eprintln!("{msg}");
                        self.last_error = Some(msg.clone());
                        self.convert_status = Some(msg);
                    } else {
                        self.open_path = result.output.display().to_string();
                        self.convert_status =
                            Some(format!("Loaded dataset: {}", result.output.display()));
                    }
                }
            }
            Ok(Err(msg)) => self.convert_status = Some(msg),
            Err(_) => self.convert_status = Some("Conversion thread panicked.".to_string()),
        }
        self.refresh_convert_log();
        if let Some(path) = self.convert_log_path.clone() {
            let is_mock_log = path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("mock_convert_log_"))
                .unwrap_or(false);
            if is_mock_log && Self::path_within_root(&self.logs_dir, &path) {
                let _ = std::fs::remove_file(&path);
                self.convert_log_path = None;
            }
        }
    }

    pub(super) fn refresh_convert_log(&mut self) {
        let Some(path) = self.convert_log_path.as_ref() else {
            return;
        };
        if let Ok(text) = std::fs::read_to_string(path) {
            self.convert_log_text = text;
        }
    }

    pub(super) fn queue_h5ad_convert(&mut self, path: &Path) {
        self.convert_input = path.display().to_string();
        self.convert_output = self
            .default_convert_output(&self.convert_input)
            .display()
            .to_string();
        self.start_convert();
    }

    pub(super) fn clamp_mock_cells(&mut self) -> u32 {
        let clamped = self.mock_cells.clamp(10_000, 10_000_000);
        if clamped != self.mock_cells {
            self.mock_cells = clamped;
        }
        clamped
    }

    pub(super) fn default_mock_paths(&self, cells: u32) -> (PathBuf, PathBuf) {
        let _ = std::fs::create_dir_all(&self.output_dir);
        let ts = chrono_like_timestamp();
        let h5ad = self
            .output_dir
            .join(format!("mock_spatial_{cells}_{ts}.h5ad"));
        let stviz = self
            .output_dir
            .join(format!("mock_spatial_{cells}_{ts}.stviz"));
        (h5ad, stviz)
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
            if name.starts_with("mock_spatial_") || name.starts_with("mock_convert_log_") {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    pub(super) fn cleanup_previous_mock(&mut self) {
        if let Some(path) = self.mock_last_h5ad.as_ref() {
            if Self::path_within_root(&self.output_dir, path) {
                let _ = std::fs::remove_file(path);
            }
        }
        if let Some(path) = self.mock_last_stviz.as_ref() {
            if Self::path_within_root(&self.output_dir, path) {
                let _ = std::fs::remove_file(path);
            }
        }
        if let Some(path) = self.mock_last_log.as_ref() {
            if Self::path_within_root(&self.logs_dir, path) {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    pub(super) fn start_mock_dataset(&mut self) {
        if self.convert_running {
            return;
        }
        self.cleanup_previous_mock();
        let cells = self.clamp_mock_cells();
        let (h5ad_path, stviz_path) = self.default_mock_paths(cells);
        self.mock_last_h5ad = Some(h5ad_path.clone());
        self.mock_last_stviz = Some(stviz_path.clone());
        self.convert_input = h5ad_path.display().to_string();
        self.convert_output = stviz_path.display().to_string();

        let script = self
            .project_dir
            .join("python")
            .join("mock_spatial_dataset.py");
        if !script.exists() {
            self.convert_status = Some(format!(
                "Mock dataset script not found: {}",
                script.display()
            ));
            self.last_error = Some(format!(
                "Mock dataset script not found: {}",
                script.display()
            ));
            return;
        }

        let (python_cmd, python_exe) = match Self::detect_python_cmd() {
            Ok(cmd) => cmd,
            Err(msg) => {
                self.convert_status = Some(msg);
                return;
            }
        };
        self.convert_last_python_exe = Some(python_exe.clone());

        let input = h5ad_path.display().to_string();
        let output = stviz_path.display().to_string();
        let script = script.to_string_lossy().to_string();
        let include_expr = self.convert_include_expr;
        let generate_only = self.convert_generate_only;
        let project_dir = self.project_dir.clone();
        let converter_venv_dir = self.converter_venv_dir.clone();
        let log_path = self
            .logs_dir
            .join(format!("mock_convert_log_{}.txt", chrono_like_timestamp()));
        let seed: u64 = rand::thread_rng().gen();

        self.convert_running = true;
        self.convert_log_path = Some(log_path.clone());
        self.convert_log_text.clear();
        self.convert_status = Some("Preparing mock dataset...".to_string());
        self.convert_handle = Some(thread::spawn(move || {
            let mut log = String::new();
            let mut append = |line: &str| {
                log.push_str(line);
                log.push('\n');
                let _ = std::fs::write(&log_path, &log);
            };

            append("Mock dataset generation started.");
            append(&format!("Requested cells: {cells}"));
            append(if include_expr {
                "Including gene expression data."
            } else {
                "Skipping gene expression data for smaller file size."
            });
            append("Preparing the converter environment...");
            append(&format!("Project dir: {}", project_dir.display()));
            append(&format!("Python cmd: {}", python_cmd));

            let venv_dir = converter_venv_dir;
            let venv_python = Self::venv_python_path(&venv_dir);
            let base_env_root = Self::python_env_root(&python_cmd);
            if !venv_python.exists() {
                append("Creating managed converter environment...");
                let mut cmd = Command::new(&python_cmd);
                cmd.arg("-m").arg("venv").arg(&venv_dir);
                Self::apply_python_env(&mut cmd, base_env_root.as_deref());
                let out = cmd.output().map_err(|e| {
                    append("Failed to create the converter environment.");
                    append(&format!("Details: {e}"));
                    format!(
                        "Failed to create virtual environment: {e}\nLog: {}",
                        log_path.display()
                    )
                })?;
                if !out.status.success() {
                    append("Environment setup output:");
                    append(&String::from_utf8_lossy(&out.stdout));
                    append("Environment setup error:");
                    append(&String::from_utf8_lossy(&out.stderr));
                    return Err(format!(
                        "Virtual environment setup failed.\nLog: {}",
                        log_path.display()
                    ));
                }
            }
            append(&format!("Venv dir: {}", venv_dir.display()));
            append(&format!("Venv python: {}", venv_python.display()));
            append(&format!("Converter runtime: {}", venv_python.display()));

            append("Updating Python packaging tools...");
            let mut bootstrap_cmd = Command::new(&venv_python);
            bootstrap_cmd
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg("--upgrade")
                .arg("pip")
                .arg("setuptools")
                .arg("wheel");
            Self::apply_python_env(&mut bootstrap_cmd, Some(&venv_dir));
            let bootstrap_out = bootstrap_cmd.output().map_err(|e| {
                append("Failed to bootstrap pip tooling.");
                append(&format!("Details: {e}"));
                format!(
                    "Failed to bootstrap pip tooling: {e}\nLog: {}",
                    log_path.display()
                )
            })?;
            if !bootstrap_out.status.success() {
                append("Bootstrap output:");
                append(&String::from_utf8_lossy(&bootstrap_out.stdout));
                append("Bootstrap error:");
                append(&String::from_utf8_lossy(&bootstrap_out.stderr));
                return Err(format!(
                    "Failed to bootstrap pip tooling.\nLog: {}",
                    log_path.display()
                ));
            }

            append("Checking converter dependencies...");
            let mut check_cmd = Command::new(&venv_python);
            check_cmd
                .arg("-c")
                .arg("import anndata, h5py, numpy, pandas, scipy");
            Self::apply_python_env(&mut check_cmd, Some(&venv_dir));
            append("Dependency check: import anndata, h5py, numpy, pandas, scipy");
            let check_out = check_cmd.output().map_err(|e| {
                append("Dependency check failed.");
                append(&format!("Details: {e}"));
                format!(
                    "Failed to check dependencies: {e}\nLog: {}",
                    log_path.display()
                )
            })?;

            if !check_out.status.success() {
                append("Installing converter dependencies (first-run setup)...");
                let requirements = project_dir.join("python").join("requirements.txt");
                let mut pip_cmd = Command::new(&venv_python);
                pip_cmd
                    .arg("-m")
                    .arg("pip")
                    .arg("install")
                    .arg("--prefer-binary");
                if requirements.exists() {
                    append(&format!(
                        "pip install --prefer-binary -r {}",
                        requirements.display()
                    ));
                    pip_cmd.arg("-r").arg(&requirements);
                } else {
                    append("requirements.txt not found; installing core converter deps.");
                    pip_cmd
                        .arg("anndata")
                        .arg("h5py")
                        .arg("numpy")
                        .arg("pandas")
                        .arg("scipy");
                }
                Self::apply_python_env(&mut pip_cmd, Some(&venv_dir));
                let pip_out = pip_cmd.output().map_err(|e| {
                    append("Dependency install failed.");
                    append(&format!("Details: {e}"));
                    format!(
                        "Failed to install dependencies: {e}\nLog: {}",
                        log_path.display()
                    )
                })?;
                if !pip_out.status.success() {
                    append("Install output:");
                    append(&String::from_utf8_lossy(&pip_out.stdout));
                    append("Install error:");
                    append(&String::from_utf8_lossy(&pip_out.stderr));
                    return Err(format!(
                        "Dependency install failed.\nLog: {}",
                        log_path.display()
                    ));
                }
                append("Dependencies installed.");
            }

            append("Generating the mock .h5ad file...");
            let mut mock_cmd = Command::new(&venv_python);
            mock_cmd
                .arg("-X")
                .arg("faulthandler")
                .arg(&script)
                .arg("--out")
                .arg(&input)
                .arg("--cells")
                .arg(cells.to_string())
                .arg("--seed")
                .arg(seed.to_string());
            if !include_expr {
                mock_cmd.arg("--no-expr");
            }
            mock_cmd.env("PYTHONFAULTHANDLER", "1");
            mock_cmd.current_dir(&project_dir);
            Self::apply_python_env(&mut mock_cmd, Some(&venv_dir));
            append(&format!(
                "Command: {} {} --out {} --cells {} --seed {}{}",
                venv_python.display(),
                script,
                input,
                cells,
                seed,
                if include_expr { "" } else { " --no-expr" }
            ));

            let out = mock_cmd.output().map_err(|e| {
                append("Mock dataset generation failed to start.");
                append(&format!("Details: {e}"));
                format!(
                    "Failed to run mock generator: {e}\nLog: {}",
                    log_path.display()
                )
            })?;
            if !out.status.success() {
                append("Mock generation output:");
                append(&String::from_utf8_lossy(&out.stdout));
                append("Mock generation error:");
                append(&String::from_utf8_lossy(&out.stderr));
                return Err(format!(
                    "Mock dataset generation failed.\nLog: {}",
                    log_path.display()
                ));
            }
            append("Mock dataset created.");

            append("Converting dataset...");
            let mut cmd = Command::new(&venv_python);
            cmd.arg("-X")
                .arg("faulthandler")
                .arg(project_dir.join("python").join("export_stviz.py"))
                .arg("--input")
                .arg(&input)
                .arg("--output")
                .arg(&output);
            if include_expr {
                cmd.arg("--include-expr");
            }
            cmd.env("PYTHONFAULTHANDLER", "1");
            cmd.current_dir(&project_dir);
            Self::apply_python_env(&mut cmd, Some(&venv_dir));
            append(&format!(
                "Command: {} export_stviz.py --input {} --output {}{}",
                venv_python.display(),
                input,
                output,
                if include_expr { " --include-expr" } else { "" }
            ));

            let out = cmd.output().map_err(|e| {
                append("Conversion failed to start.");
                append(&format!("Details: {e}"));
                format!("Failed to run exporter: {e}\nLog: {}", log_path.display())
            })?;
            if !out.status.success() {
                append("Conversion output:");
                append(&String::from_utf8_lossy(&out.stdout));
                append("Conversion error:");
                append(&String::from_utf8_lossy(&out.stderr));
                return Err(format!("Conversion failed.\nLog: {}", log_path.display()));
            }
            append("Conversion completed.");
            let stdout = String::from_utf8_lossy(&out.stdout);
            Ok(ConvertResult {
                msg: format!("Mock dataset ready.\n{}", stdout.trim()),
                output: PathBuf::from(output),
                load_after: !generate_only,
                python_exe: Some(venv_python.display().to_string()),
            })
        }));
    }

    pub(super) fn start_convert(&mut self) {
        if self.convert_running {
            return;
        }
        let input = self.convert_input.trim();
        if input.is_empty() {
            self.convert_status = Some("Input .h5ad path is empty.".to_string());
            return;
        }
        if self.convert_output.trim().is_empty() {
            let out = self.default_convert_output(input);
            self.convert_output = out.display().to_string();
        }
        let output = self.convert_output.trim();
        if output.is_empty() {
            self.convert_status = Some("Output .stviz path is empty.".to_string());
            return;
        }

        let script = self.project_dir.join("python").join("export_stviz.py");
        if !script.exists() {
            self.convert_status = Some(format!("Exporter not found: {}", script.display()));
            return;
        }

        let (python_cmd, python_exe) = match Self::detect_python_cmd() {
            Ok(cmd) => cmd,
            Err(msg) => {
                self.convert_status = Some(msg);
                return;
            }
        };
        self.convert_last_python_exe = Some(python_exe.clone());
        let input = input.to_string();
        let output = output.to_string();
        let script = script.to_string_lossy().to_string();
        let include_expr = self.convert_include_expr;
        let generate_only = self.convert_generate_only;
        let project_dir = self.project_dir.clone();
        let converter_venv_dir = self.converter_venv_dir.clone();
        let log_path = self
            .logs_dir
            .join(format!("convert_log_{}.txt", chrono_like_timestamp()));
        self.convert_running = true;
        self.convert_log_path = Some(log_path.clone());
        self.convert_log_text.clear();
        self.convert_status = Some("Checking converter dependencies...".to_string());
        self.convert_handle = Some(thread::spawn(move || {
            let mut log = String::new();
            let mut append = |line: &str| {
                log.push_str(line);
                log.push('\n');
                let _ = std::fs::write(&log_path, &log);
            };

            append("Conversion started.");
            append(if include_expr {
                "Including gene expression data."
            } else {
                "Skipping gene expression data for smaller file size."
            });
            append("Preparing the converter environment...");
            append(&format!("Input file: {input}"));
            append(&format!("Output file: {output}"));
            append(&format!("Project dir: {}", project_dir.display()));
            append(&format!("Python cmd: {}", python_cmd));

            let venv_dir = converter_venv_dir;
            let venv_python = Self::venv_python_path(&venv_dir);
            let base_env_root = Self::python_env_root(&python_cmd);
            if !venv_python.exists() {
                append("Creating managed converter environment...");
                let mut cmd = Command::new(&python_cmd);
                cmd.arg("-m").arg("venv").arg(&venv_dir);
                Self::apply_python_env(&mut cmd, base_env_root.as_deref());
                let out = cmd.output().map_err(|e| {
                    append("Failed to create the converter environment.");
                    append(&format!("Details: {e}"));
                    format!(
                        "Failed to create virtual environment: {e}\nLog: {}",
                        log_path.display()
                    )
                })?;
                if !out.status.success() {
                    append("Environment setup output:");
                    append(&String::from_utf8_lossy(&out.stdout));
                    append("Environment setup error:");
                    append(&String::from_utf8_lossy(&out.stderr));
                    return Err(format!(
                        "Virtual environment setup failed.\nLog: {}",
                        log_path.display()
                    ));
                }
            }
            append(&format!("Venv dir: {}", venv_dir.display()));
            append(&format!("Venv python: {}", venv_python.display()));
            append(&format!("Converter runtime: {}", venv_python.display()));

            append("Updating Python packaging tools...");
            let mut bootstrap_cmd = Command::new(&venv_python);
            bootstrap_cmd
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg("--upgrade")
                .arg("pip")
                .arg("setuptools")
                .arg("wheel");
            Self::apply_python_env(&mut bootstrap_cmd, Some(&venv_dir));
            let bootstrap_out = bootstrap_cmd.output().map_err(|e| {
                append("Failed to bootstrap pip tooling.");
                append(&format!("Details: {e}"));
                format!(
                    "Failed to bootstrap pip tooling: {e}\nLog: {}",
                    log_path.display()
                )
            })?;
            if !bootstrap_out.status.success() {
                append("Bootstrap output:");
                append(&String::from_utf8_lossy(&bootstrap_out.stdout));
                append("Bootstrap error:");
                append(&String::from_utf8_lossy(&bootstrap_out.stderr));
                return Err(format!(
                    "Failed to bootstrap pip tooling.\nLog: {}",
                    log_path.display()
                ));
            }

            append("Checking converter dependencies...");
            let mut check_cmd = Command::new(&venv_python);
            check_cmd
                .arg("-c")
                .arg("import anndata, h5py, numpy, pandas, scipy");
            Self::apply_python_env(&mut check_cmd, Some(&venv_dir));
            append("Dependency check: import anndata, h5py, numpy, pandas, scipy");
            let check_out = check_cmd.output().map_err(|e| {
                append("Dependency check failed.");
                append(&format!("Details: {e}"));
                format!(
                    "Failed to check dependencies: {e}\nLog: {}",
                    log_path.display()
                )
            })?;

            if !check_out.status.success() {
                append("Installing converter dependencies (first-run setup)...");
                let requirements = project_dir.join("python").join("requirements.txt");
                let mut pip_cmd = Command::new(&venv_python);
                pip_cmd
                    .arg("-m")
                    .arg("pip")
                    .arg("install")
                    .arg("--prefer-binary");
                if requirements.exists() {
                    append(&format!(
                        "pip install --prefer-binary -r {}",
                        requirements.display()
                    ));
                    pip_cmd.arg("-r").arg(&requirements);
                } else {
                    append("requirements.txt not found; installing core converter deps.");
                    pip_cmd
                        .arg("anndata")
                        .arg("h5py")
                        .arg("numpy")
                        .arg("pandas")
                        .arg("scipy");
                }
                Self::apply_python_env(&mut pip_cmd, Some(&venv_dir));
                let pip_out = pip_cmd.output().map_err(|e| {
                    append("Dependency install failed.");
                    append(&format!("Details: {e}"));
                    format!(
                        "Failed to install dependencies: {e}\nLog: {}",
                        log_path.display()
                    )
                })?;
                if !pip_out.status.success() {
                    append("Install output:");
                    append(&String::from_utf8_lossy(&pip_out.stdout));
                    append("Install error:");
                    append(&String::from_utf8_lossy(&pip_out.stderr));
                    return Err(format!(
                        "Dependency install failed.\nLog: {}",
                        log_path.display()
                    ));
                }
                append("Dependencies installed.");
            }

            append("Converting dataset...");
            let mut cmd = Command::new(&venv_python);
            cmd.arg("-X")
                .arg("faulthandler")
                .arg(script)
                .arg("--input")
                .arg(&input)
                .arg("--output")
                .arg(&output);
            if include_expr {
                cmd.arg("--include-expr");
            }
            cmd.env("PYTHONFAULTHANDLER", "1");
            cmd.current_dir(&project_dir);
            Self::apply_python_env(&mut cmd, Some(&venv_dir));
            append(&format!(
                "Command: {} export_stviz.py --input {} --output {}{}",
                venv_python.display(),
                input,
                output,
                if include_expr { " --include-expr" } else { "" }
            ));

            let out = cmd.output().map_err(|e| {
                append("Conversion failed to start.");
                append(&format!("Details: {e}"));
                format!("Failed to run exporter: {e}\nLog: {}", log_path.display())
            })?;
            if !out.status.success() {
                append("Conversion output:");
                append(&String::from_utf8_lossy(&out.stdout));
                append("Conversion error:");
                append(&String::from_utf8_lossy(&out.stderr));
                return Err(format!("Conversion failed.\nLog: {}", log_path.display()));
            }
            append("Conversion completed.");
            let stdout = String::from_utf8_lossy(&out.stdout);
            Ok(ConvertResult {
                msg: format!("Conversion completed.\n{}", stdout.trim()),
                output: PathBuf::from(output),
                load_after: !generate_only,
                python_exe: Some(venv_python.display().to_string()),
            })
        }));
    }
}
