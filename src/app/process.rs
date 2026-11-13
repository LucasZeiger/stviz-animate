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

pub(super) fn wait_cancellable(
    child: &mut Child,
    cancel: &AtomicBool,
) -> Result<ExitStatus, RunError> {
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

/// Run a command, passing each line of stdout/stderr to `on_line` as it is
/// printed. Kills the process if `cancel` is set.
pub(super) fn run_streaming(
    cmd: &mut Command,
    cancel: &AtomicBool,
    on_line: &mut dyn FnMut(&str),
) -> Result<ExitStatus, RunError> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| RunError::Failed(e.to_string()))?;
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
            let _ = child.kill();
            let _ = child.wait();
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::Arc;

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
