//! Running subprocesses from worker threads: cancellable, with output either
//! captured or streamed line by line.

use std::{
    io::{BufRead, BufReader, Read},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::JoinHandle,
    time::Duration,
};

/// How often a worker checks a running subprocess for exit or cancellation.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

pub(super) enum RunError {
    Cancelled,
    Failed(String),
}

impl From<String> for RunError {
    fn from(msg: String) -> Self {
        Self::Failed(msg)
    }
}

pub(super) struct ProcessOutput {
    pub(super) status: ExitStatus,
    pub(super) stdout: Vec<u8>,
    pub(super) stderr: Vec<u8>,
}

/// Like `Command::output`, but kills the process if `cancel` is set.
pub(super) fn run_cancellable(
    cmd: &mut Command,
    cancel: &AtomicBool,
) -> Result<ProcessOutput, RunError> {
    let mut child = spawn_piped(cmd)?;
    // Drain both pipes on their own threads so a chatty process can't fill a
    // pipe buffer and stall while we wait for it.
    let stdout = child.stdout.take().map(read_all_in_background);
    let stderr = child.stderr.take().map(read_all_in_background);
    // After a cancel, don't wait for the readers: a grandchild that survived
    // could keep the pipes open indefinitely.
    let status = wait_cancellable(&mut child, cancel)?;
    let collect =
        |h: Option<JoinHandle<Vec<u8>>>| h.and_then(|h| h.join().ok()).unwrap_or_default();
    let stdout = collect(stdout);
    let stderr = collect(stderr);
    Ok(ProcessOutput {
        status,
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

pub(super) fn wait_cancellable(
    child: &mut Child,
    cancel: &AtomicBool,
) -> Result<ExitStatus, RunError> {
    loop {
        if cancel.load(Ordering::Relaxed) {
            kill_tree(child);
            return Err(RunError::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) => std::thread::sleep(POLL_INTERVAL),
            Err(e) => return Err(RunError::Failed(format!("wait failed: {e}"))),
        }
    }
}

/// Spawn with piped output, in a way that lets [`kill_tree`] stop the
/// process together with anything it starts.
fn spawn_piped(cmd: &mut Command) -> Result<Child, RunError> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Own process group, so the whole group can be signalled on cancel.
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(cmd, 0);
    cmd.spawn().map_err(|e| RunError::Failed(e.to_string()))
}

/// Kill a subprocess and everything it started. On Windows a venv's
/// `python.exe` is a small launcher that runs the real interpreter as a child
/// process, and pip and shells start children too, so killing only the
/// direct child would leave the work running and its output pipes open.
pub(super) fn kill_tree(child: &mut Child) {
    // Processes from `spawn_piped` lead their own process group; signal it.
    // For other children this fails harmlessly and `kill` below handles them.
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{}", child.id())])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &child.id().to_string()])
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Run a command, passing each line of stdout/stderr to `on_line` as it is
/// printed. Kills the process if `cancel` is set.
pub(super) fn run_streaming(
    cmd: &mut Command,
    cancel: &AtomicBool,
    on_line: &mut dyn FnMut(&str),
) -> Result<ExitStatus, RunError> {
    let mut child = spawn_piped(cmd)?;
    let (tx, rx) = mpsc::channel::<String>();
    let readers = [
        child.stdout.take().map(|p| forward_lines(p, tx.clone())),
        child.stderr.take().map(|p| forward_lines(p, tx.clone())),
    ];
    drop(tx);
    let status = loop {
        for line in rx.try_iter() {
            on_line(&line);
        }
        if cancel.load(Ordering::Relaxed) {
            kill_tree(&mut child);
            return Err(RunError::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(POLL_INTERVAL),
            Err(e) => return Err(RunError::Failed(format!("wait failed: {e}"))),
        }
    };
    for reader in readers.into_iter().flatten() {
        let _ = reader.join();
    }
    for line in rx.try_iter() {
        on_line(&line);
    }
    Ok(status)
}

/// Forward a pipe line by line. Carriage returns (progress bars) also end a line.
fn forward_lines<R: Read + Send + 'static>(pipe: R, tx: mpsc::Sender<String>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(pipe);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let text = String::from_utf8_lossy(&buf);
            for part in text.split(['\r', '\n']) {
                let part = part.trim_end();
                if !part.is_empty() && tx.send(part.to_string()).is_err() {
                    return;
                }
            }
        }
    })
}

#[cfg(test)]
mod tree_tests {
    use super::*;
    use std::sync::Arc;

    /// Cancelling must stop a process that is itself waiting on a child
    /// (like a venv's python launcher on Windows), and must not hang on the
    /// output pipes the child inherited.
    fn shell_with_child() -> Command {
        if cfg!(windows) {
            let mut cmd = Command::new("cmd");
            cmd.args(["/C", "ping -n 60 127.0.0.1 & echo done"]);
            cmd
        } else {
            let mut cmd = Command::new("sh");
            cmd.args(["-c", "sleep 60; echo done"]);
            cmd
        }
    }

    fn cancel_after(ms: u64) -> (Arc<AtomicBool>, std::thread::JoinHandle<()>) {
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let setter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            flag.store(true, Ordering::Relaxed);
        });
        (cancel, setter)
    }

    #[test]
    fn cancel_stops_a_process_tree_promptly() {
        let (cancel, setter) = cancel_after(500);
        let start = std::time::Instant::now();
        let result = run_cancellable(&mut shell_with_child(), &cancel);
        setter.join().unwrap();
        assert!(matches!(result, Err(RunError::Cancelled)));
        assert!(start.elapsed() < Duration::from_secs(20));

        let (cancel, setter) = cancel_after(500);
        let start = std::time::Instant::now();
        let result = run_streaming(&mut shell_with_child(), &cancel, &mut |_| {});
        setter.join().unwrap();
        assert!(matches!(result, Err(RunError::Cancelled)));
        assert!(start.elapsed() < Duration::from_secs(20));
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// The grandchild itself must be gone after a cancel, not just abandoned.
    #[test]
    fn cancel_kills_grandchildren() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "sleep 60 & echo $!; wait"]);
        let mut grandchild = None;
        let result = run_streaming(&mut cmd, &cancel, &mut |line| {
            grandchild = line.trim().parse::<u32>().ok();
            cancel.store(true, Ordering::Relaxed);
        });
        assert!(matches!(result, Err(RunError::Cancelled)));
        let pid = grandchild.expect("shell should print the child pid");
        let mut alive = true;
        for _ in 0..50 {
            alive = Command::new("kill")
                .args(["-0", &pid.to_string()])
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
            if !alive {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(!alive, "grandchild {pid} survived the cancel");
    }

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
    fn run_streaming_forwards_lines_from_both_streams() {
        let cancel = AtomicBool::new(false);
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "echo one; echo two >&2; printf 'a\\rb\\n'; exit 2"]);
        let mut lines = Vec::new();
        let Ok(status) = run_streaming(&mut cmd, &cancel, &mut |l| lines.push(l.to_string()))
        else {
            panic!("sh should run");
        };
        assert_eq!(status.code(), Some(2));
        lines.sort();
        assert_eq!(lines, vec!["a", "b", "one", "two"]);
    }
}
