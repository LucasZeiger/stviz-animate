//! Undo and redo of view edits.
//!
//! The history does not hook into each control. At quiet moments (no button
//! held, no text field focused) it captures the view the same way saved
//! scenes do ([`WorkspaceView`]), plus the lasso selection and the scene
//! list, and records a step whenever that differs from the last capture. A
//! slider drag or a typed title therefore becomes one step, and new controls
//! are covered without extra code.
//!
//! Panning, zooming, the playback position and UI scale/theme are not steps:
//! they are left out of the comparison and kept as they are when undoing,
//! except when the step loaded a scene, which also moved the camera.

use super::{workspace::WorkspaceView, StvizApp};
use eframe::egui;
use std::sync::Arc;

/// Steps kept in each direction.
const MAX_STEPS: usize = 100;

/// Everything undo restores.
#[derive(Clone)]
pub(super) struct Snapshot {
    view: WorkspaceView,
    selection_version: u64,
    selection: Arc<Vec<u32>>,
    scenes_version: u64,
    scenes: Arc<Vec<serde_json::Value>>,
}

impl Snapshot {
    fn same_edits(&self, other: &Self) -> bool {
        self.selection_version == other.selection_version
            && self.scenes_version == other.scenes_version
            && self.view.same_edits(&other.view)
    }
}

/// One recorded change: the state before and after it.
#[derive(Clone)]
struct Step<S> {
    before: S,
    after: S,
    /// The change moved the camera on purpose (a scene was loaded), so undo
    /// and redo move it too.
    moves_camera: bool,
}

/// Undo and redo stacks, independent of what a state holds.
pub(super) struct Stacks<S> {
    undo: Vec<Step<S>>,
    redo: Vec<Step<S>>,
}

impl<S> Default for Stacks<S> {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }
}

impl<S> Stacks<S> {
    fn push(&mut self, before: S, after: S, moves_camera: bool) {
        if self.undo.len() == MAX_STEPS {
            self.undo.remove(0);
        }
        self.undo.push(Step {
            before,
            after,
            moves_camera,
        });
        self.redo.clear();
    }

    /// Take the step to undo; it moves to the redo stack.
    fn undo(&mut self) -> Option<&Step<S>> {
        let step = self.undo.pop()?;
        self.redo.push(step);
        self.redo.last()
    }

    /// Take the step to redo; it moves back to the undo stack.
    fn redo(&mut self) -> Option<&Step<S>> {
        let step = self.redo.pop()?;
        self.undo.push(step);
        self.undo.last()
    }

    fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Request {
    Undo,
    Redo,
}

#[derive(Default)]
pub(super) struct History {
    stacks: Stacks<Snapshot>,
    /// State at the last capture, with the camera kept current.
    last: Option<Snapshot>,
    /// Dataset the history belongs to (pointer identity).
    dataset: usize,
    /// Input arrived since the last capture.
    dirty: bool,
    /// The change being made loaded a scene.
    view_jump: bool,
    request: Option<Request>,
}

impl History {
    pub(super) fn note_view_jump(&mut self) {
        self.view_jump = true;
    }

    pub(super) fn can_undo(&self) -> bool {
        self.stacks.can_undo()
    }

    pub(super) fn can_redo(&self) -> bool {
        self.stacks.can_redo()
    }

    pub(super) fn request_undo(&mut self) {
        self.request = Some(Request::Undo);
    }

    pub(super) fn request_redo(&mut self) {
        self.request = Some(Request::Redo);
    }
}

impl StvizApp {
    fn capture_snapshot(&self, last: Option<&Snapshot>) -> Option<Snapshot> {
        let view = self.build_view().ok()?;
        let selection_version = self.selection.version();
        let selection = match last {
            Some(l) if l.selection_version == selection_version => l.selection.clone(),
            _ => Arc::new(self.selection.selected().to_vec()),
        };
        let scenes_version = self.scenes.version();
        let scenes = match last {
            Some(l) if l.scenes_version == scenes_version => l.scenes.clone(),
            _ => Arc::new(self.scenes.raw().to_vec()),
        };
        Some(Snapshot {
            view,
            selection_version,
            selection,
            scenes_version,
            scenes,
        })
    }

    /// Ctrl/Cmd+Z, Ctrl/Cmd+Shift+Z and Ctrl+Y. Text fields keep their own
    /// undo: `handle_hotkeys` does not run while one has focus.
    pub(super) fn handle_undo_hotkeys(&mut self, ctx: &egui::Context) {
        use egui::{Key, KeyboardShortcut, Modifiers};
        let redo_shift = KeyboardShortcut::new(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z);
        let undo = KeyboardShortcut::new(Modifiers::COMMAND, Key::Z);
        let redo_y = KeyboardShortcut::new(Modifiers::CTRL, Key::Y);
        ctx.input_mut(|i| {
            // Shift+Z first: the plain shortcut also matches with Shift held.
            if i.consume_shortcut(&redo_shift) || i.consume_shortcut(&redo_y) {
                self.history.request_redo();
            } else if i.consume_shortcut(&undo) {
                self.history.request_undo();
            }
        });
    }

    /// Run once per frame, after the UI: record changes made this frame and
    /// carry out a requested undo or redo.
    pub(super) fn update_history(&mut self, ctx: &egui::Context) {
        let ds_ptr = self
            .dataset
            .as_ref()
            .map_or(0, |ds| Arc::as_ptr(ds) as usize);
        if ds_ptr != self.history.dataset {
            self.history = History {
                dataset: ds_ptr,
                ..History::default()
            };
        }
        if ds_ptr == 0 {
            return;
        }
        if ctx.input(|i| !i.events.is_empty() || i.pointer.any_released()) {
            self.history.dirty = true;
        }
        let busy = self.scenes_busy() || self.exporting_loop;
        let request = self.history.request.take();
        if busy {
            if request.is_some() {
                self.last_error =
                    Some("Undo and redo wait until the conversion, load or export ends.".into());
            }
            return;
        }
        let settling = ctx.input(|i| i.pointer.any_down()) || ctx.wants_keyboard_input();
        if self.history.last.is_none() || (self.history.dirty && !settling) || request.is_some() {
            self.record_changes();
        }
        if let Some(request) = request {
            self.apply_history(request);
        }
    }

    /// Compare the current state with the last capture and record a step if
    /// anything undoable changed.
    fn record_changes(&mut self) {
        self.history.dirty = false;
        let Some(now) = self.capture_snapshot(self.history.last.as_ref()) else {
            return;
        };
        let Some(last) = self.history.last.take() else {
            self.history.last = Some(now);
            return;
        };
        if !now.same_edits(&last) {
            let moves_camera = std::mem::take(&mut self.history.view_jump);
            self.history.stacks.push(last, now.clone(), moves_camera);
        }
        self.history.last = Some(now);
    }

    fn apply_history(&mut self, request: Request) {
        let step = match request {
            Request::Undo => self.history.stacks.undo(),
            Request::Redo => self.history.stacks.redo(),
        };
        let Some(step) = step else {
            return;
        };
        let (target, moves_camera) = match request {
            Request::Undo => (step.before.clone(), step.moves_camera),
            Request::Redo => (step.after.clone(), step.moves_camera),
        };
        if let Err(e) = self.restore_snapshot(&target, moves_camera) {
            self.last_error = Some(format!(
                "{} failed: {e:#}",
                if request == Request::Undo {
                    "Undo"
                } else {
                    "Redo"
                }
            ));
        }
        // Start comparing from what is on screen now, which may differ from
        // the snapshot where the dataset could not take part of it.
        self.history.last = self.capture_snapshot(None);
        self.history.view_jump = false;
        self.history.dirty = false;
    }

    fn restore_snapshot(&mut self, target: &Snapshot, moves_camera: bool) -> anyhow::Result<()> {
        let current = self
            .capture_snapshot(self.history.last.as_ref())
            .ok_or_else(|| anyhow::anyhow!("nothing to restore"))?;
        if target.scenes_version != current.scenes_version
            && target.scenes.as_slice() != current.scenes.as_slice()
        {
            self.store_scenes(target.scenes.to_vec())?;
        }
        let view_changed = !target.view.same_edits(&current.view);
        if view_changed || moves_camera {
            let (Some(ds), Some(path)) = (self.dataset.clone(), self.dataset_path.clone()) else {
                anyhow::bail!("no dataset loaded");
            };
            let camera = self.camera;
            let t = self.t;
            self.apply_view(ds, &path, target.view.clone(), false)?;
            if !moves_camera {
                self.camera = camera;
                self.t = t;
            }
        }
        // Reloading the view starts a fresh selection, so restore it whenever
        // the view was reloaded too.
        if view_changed || moves_camera || target.selection_version != current.selection_version {
            self.restore_selection(&target.selection);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_and_redo_walk_the_steps() {
        let mut s = Stacks::<u32>::default();
        s.push(0, 1, false);
        s.push(1, 2, true);
        assert!(s.can_undo() && !s.can_redo());
        let step = s.undo().unwrap();
        assert_eq!((step.before, step.moves_camera), (1, true));
        assert_eq!(s.undo().unwrap().before, 0);
        assert!(!s.can_undo() && s.can_redo());
        assert_eq!(s.redo().unwrap().after, 1);
        // A new change drops what could be redone.
        s.push(1, 5, false);
        assert!(!s.can_redo());
        assert_eq!(s.undo().unwrap().before, 1);
    }

    #[test]
    fn history_is_capped() {
        let mut s = Stacks::<usize>::default();
        for i in 0..MAX_STEPS + 10 {
            s.push(i, i + 1, false);
        }
        let mut n = 0;
        let mut oldest = None;
        while let Some(step) = s.undo() {
            n += 1;
            oldest = Some(step.before);
        }
        assert_eq!(n, MAX_STEPS);
        assert_eq!(oldest, Some(10));
    }
}
