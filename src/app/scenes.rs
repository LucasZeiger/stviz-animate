//! Named scenes saved inside the .stviz file itself.
//!
//! A scene is the same view a workspace restores (timeline, colours and
//! palettes, filters, camera, labels, playback and export settings), stored
//! under a name in the file's scenes block (see `data::write_scenes`), so a
//! single .stviz reproduces a view or video. Saving or deleting a scene only
//! rewrites that block, never the data arrays.

use super::{workspace::WorkspaceView, StvizApp};
use crate::data::Dataset;
use eframe::egui;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const MAX_SCENE_NAME_CHARS: usize = 80;

#[derive(Clone, Serialize, Deserialize)]
struct Scene {
    name: String,
    /// App version that saved the scene, for troubleshooting.
    #[serde(default)]
    saved_with: Option<String>,
    view: WorkspaceView,
}

/// Scenes of the open dataset, mirroring what is in its file.
#[derive(Default)]
pub(super) struct SceneState {
    /// Raw entries, so ones this version can't read are kept when saving.
    raw: Vec<serde_json::Value>,
    /// Whether each raw entry can be loaded by this version.
    readable: Vec<bool>,
    /// Problem reading the file's scenes, shown above the list.
    warning: Option<String>,
    name_input: String,
    status: Option<String>,
    /// Index of the scene whose Delete button was pressed once.
    confirm_delete: Option<usize>,
}

impl SceneState {
    pub(super) fn from_dataset(ds: &Dataset) -> Self {
        let mut state = Self {
            warning: ds.saved_scenes.warning.clone(),
            ..Self::default()
        };
        state.set_raw(ds.saved_scenes.scenes.clone());
        state
    }

    fn set_raw(&mut self, raw: Vec<serde_json::Value>) {
        self.readable = raw.iter().map(|r| Scene::deserialize(r).is_ok()).collect();
        self.raw = raw;
        self.confirm_delete = None;
    }
}

fn scene_name(raw: &serde_json::Value) -> &str {
    raw.get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("(unnamed)")
}

fn clean_name(name: &str) -> String {
    name.trim().chars().take(MAX_SCENE_NAME_CHARS).collect()
}

/// `scenes` with `scene` added, replacing a scene of the same name.
fn upsert(scenes: &[serde_json::Value], scene: serde_json::Value) -> Vec<serde_json::Value> {
    let name = scene_name(&scene).to_string();
    let mut out = scenes.to_vec();
    match out.iter().position(|s| scene_name(s) == name) {
        Some(i) => out[i] = scene,
        None => out.push(scene),
    }
    out
}

impl StvizApp {
    fn scenes_busy(&self) -> bool {
        self.convert_running() || self.dataset_load.is_some() || self.export_busy()
    }

    /// Write `scenes` to the dataset's file; on success they become the list.
    fn store_scenes(&mut self, scenes: Vec<serde_json::Value>) -> anyhow::Result<()> {
        let (Some(ds), Some(path)) = (self.dataset.as_ref(), self.dataset_path.as_ref()) else {
            anyhow::bail!("no dataset loaded");
        };
        ds.write_scenes(path, &scenes)?;
        self.scenes.set_raw(scenes);
        self.scenes.warning = None;
        Ok(())
    }

    fn save_scene(&mut self, name: &str) -> anyhow::Result<()> {
        let name = clean_name(name);
        if name.is_empty() {
            anyhow::bail!("enter a name for the scene");
        }
        let scene = Scene {
            name: name.clone(),
            saved_with: Some(env!("CARGO_PKG_VERSION").to_string()),
            view: self.build_view()?,
        };
        let scenes = upsert(&self.scenes.raw, serde_json::to_value(scene)?);
        self.store_scenes(scenes)?;
        self.scenes.status = Some(format!("Saved scene \"{name}\"."));
        Ok(())
    }

    fn load_scene(&mut self, idx: usize) -> anyhow::Result<()> {
        if self.scenes_busy() {
            anyhow::bail!("wait for the running conversion, load or export to finish");
        }
        let raw = self
            .scenes
            .raw
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("scene not found"))?;
        let scene: Scene = serde_json::from_value(raw.clone())
            .map_err(|_| anyhow::anyhow!("this scene can't be opened by this version"))?;
        let (Some(ds), Some(path)) = (self.dataset.clone(), self.dataset_path.clone()) else {
            anyhow::bail!("no dataset loaded");
        };
        self.playing = false;
        self.apply_view(ds, &path, scene.view, false)?;
        self.scenes.name_input = scene.name.clone();
        self.scenes.status = Some(format!("Loaded scene \"{}\".", scene.name));
        Ok(())
    }

    fn delete_scene(&mut self, idx: usize) -> anyhow::Result<()> {
        if idx >= self.scenes.raw.len() {
            anyhow::bail!("scene not found");
        }
        let mut scenes = self.scenes.raw.clone();
        let removed = scenes.remove(idx);
        self.store_scenes(scenes)?;
        self.scenes.status = Some(format!("Deleted scene \"{}\".", scene_name(&removed)));
        Ok(())
    }

    /// Reset the scene list for a newly opened dataset. Re-installing the same
    /// dataset (as loading a scene does) keeps the list, which may have changed
    /// since the file was opened.
    pub(super) fn reset_scenes_for(&mut self, ds: &Arc<Dataset>) {
        if self
            .dataset
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, ds))
        {
            return;
        }
        self.scenes = SceneState::from_dataset(ds);
    }

    pub(super) fn ui_scenes_section(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Saved scenes").on_hover_text(
            "Save the timeline, colours, filters, camera, labels and export settings \
             into the .stviz file under a name, and bring them back later. Anyone who \
             opens the file gets the same scenes.",
        );
        if let Some(w) = &self.scenes.warning {
            ui.colored_label(egui::Color32::from_rgb(230, 160, 60), w);
        }

        let name = clean_name(&self.scenes.name_input);
        let exists = self.scenes.raw.iter().any(|s| scene_name(s) == name);
        ui.horizontal(|ui| {
            let edit = ui.add(
                egui::TextEdit::singleline(&mut self.scenes.name_input)
                    .hint_text("Scene name")
                    .char_limit(MAX_SCENE_NAME_CHARS)
                    .desired_width(150.0),
            );
            let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let label = if exists { "Update scene" } else { "Save scene" };
            let button = ui
                .add_enabled(!name.is_empty(), egui::Button::new(label))
                .on_hover_text(if exists {
                    "Replace the saved scene of this name with the current view."
                } else {
                    "Save the current view into the .stviz file."
                });
            if button.clicked() || (enter && !name.is_empty()) {
                if let Err(e) = self.save_scene(&name) {
                    self.scenes.status = Some(format!("Scene not saved: {e:#}"));
                }
            }
        });

        if self.scenes.raw.is_empty() {
            ui.small("No saved scenes in this file.");
        }
        let mut action: Option<(usize, bool)> = None; // (index, is_delete)
        for (i, raw) in self.scenes.raw.iter().enumerate() {
            let readable = self.scenes.readable.get(i).copied().unwrap_or(false);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(readable, egui::Button::new("Load"))
                    .on_disabled_hover_text("This scene can't be opened by this version.")
                    .clicked()
                {
                    action = Some((i, false));
                }
                let confirming = self.scenes.confirm_delete == Some(i);
                let delete = if confirming {
                    egui::Button::new("Confirm delete")
                } else {
                    egui::Button::new("Delete")
                };
                if ui
                    .add(delete)
                    .on_hover_text("Remove this scene from the file.")
                    .clicked()
                {
                    if confirming {
                        action = Some((i, true));
                    } else {
                        self.scenes.confirm_delete = Some(i);
                    }
                }
                ui.label(scene_name(raw));
            });
        }
        match action {
            Some((i, false)) => {
                if let Err(e) = self.load_scene(i) {
                    self.scenes.status = Some(format!("Scene not loaded: {e:#}"));
                }
            }
            Some((i, true)) => {
                if let Err(e) = self.delete_scene(i) {
                    self.scenes.status = Some(format!("Scene not deleted: {e:#}"));
                }
            }
            None => {}
        }
        if let Some(status) = &self.scenes.status {
            ui.small(status);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saving_a_scene_with_an_existing_name_replaces_it() {
        let a = serde_json::json!({"name": "A", "view": 1});
        let b = serde_json::json!({"name": "B", "view": 2});
        let a2 = serde_json::json!({"name": "A", "view": 3});
        let scenes = upsert(&upsert(&[a], b), a2.clone());
        assert_eq!(scenes.len(), 2);
        assert_eq!(scenes[0], a2);
        assert_eq!(scene_name(&scenes[1]), "B");
        assert_eq!(clean_name("  x  "), "x");
        assert_eq!(clean_name(&"y".repeat(200)).len(), MAX_SCENE_NAME_CHARS);
    }
}
