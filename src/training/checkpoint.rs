// Copyright (C) 2024 Bellande Artificial Intelligence Computer Vision Research Innovation Center, Ronaldson Bellande

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use crate::core::{device::Device, dtype::DataType};
use crate::core::{error::BellandeError, tensor::Tensor};
use crate::models::models::Model;
use crate::training::callbacks::Callback;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum CheckpointMode {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy)]
pub enum SaveFormat {
    Json,
    Binary,
}

pub struct ModelCheckpoint {
    filepath: String,
    monitor: String,
    save_best_only: bool,
    save_weights_only: bool,
    mode: CheckpointMode,
    best_value: f32,
    model: Option<Box<dyn Model>>,
    save_format: SaveFormat,
    verbose: bool,
    keep_best_n: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct CheckpointMetadata {
    epoch: usize,
    best_value: f32,
    monitor: String,
    mode: CheckpointMode,
    metrics: HashMap<String, f32>,
}

impl ModelCheckpoint {
    pub fn new(
        filepath: String,
        monitor: String,
        save_best_only: bool,
        save_weights_only: bool,
        mode: CheckpointMode,
    ) -> Self {
        ModelCheckpoint {
            filepath,
            monitor,
            save_best_only,
            save_weights_only,
            mode,
            best_value: match mode {
                CheckpointMode::Min => f32::INFINITY,
                CheckpointMode::Max => f32::NEG_INFINITY,
            },
            model: None,
            save_format: SaveFormat::Binary,
            verbose: true,
            keep_best_n: None,
        }
    }

    pub fn with_model(mut self, model: Box<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_save_format(mut self, format: SaveFormat) -> Self {
        self.save_format = format;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn with_keep_best_n(mut self, n: usize) -> Self {
        self.keep_best_n = Some(n);
        self
    }

    fn is_better(&self, current: f32) -> bool {
        match self.mode {
            CheckpointMode::Min => current < self.best_value,
            CheckpointMode::Max => current > self.best_value,
        }
    }

    fn save_checkpoint(
        &mut self,
        filepath: &Path,
        epoch: usize,
        metrics: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        // Create directory first
        if let Some(parent) = filepath.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to create directory: {}", e))
            })?;
        }

        // Extract values we need before borrowing self
        let save_weights_only = self.save_weights_only;
        let save_format = self.save_format;
        let verbose = self.verbose;

        // Get model reference once
        if let Some(model) = self.model.as_ref() {
            // Save model/weights without borrowing self again
            if save_weights_only {
                save_model_weights(model.as_ref(), filepath, save_format)?;
            } else {
                save_model_state(model.as_ref(), filepath, save_format)?;
            }

            let metadata = CheckpointMetadata {
                epoch,
                best_value: self.best_value,
                monitor: self.monitor.clone(),
                mode: self.mode,
                metrics: metrics.clone(),
            };

            let metadata_path = filepath.with_extension("meta.json");
            let file = File::create(metadata_path).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to create metadata file: {}", e))
            })?;

            serde_json::to_writer_pretty(file, &metadata).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to write metadata: {}", e))
            })?;

            if verbose {
                println!("Saved checkpoint to {}", filepath.display());
            }
        }
        Ok(())
    }

    fn cleanup_old_checkpoints(&mut self, keep_best_n: usize) -> Result<(), BellandeError> {
        let meta_pattern = self.filepath.replace("{epoch}", "*").replace("{val}", "*");
        let meta_pattern = format!("{}.meta.json", meta_pattern);

        let mut checkpoints: Vec<_> = glob::glob(&meta_pattern)
            .map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to read checkpoint directory: {}", e))
            })?
            .filter_map(Result::ok)
            .filter_map(|path| {
                if let Ok(file) = File::open(&path) {
                    if let Ok(metadata) = serde_json::from_reader::<_, CheckpointMetadata>(file) {
                        return Some((path, metadata));
                    }
                }
                None
            })
            .collect();

        checkpoints.sort_by(|a, b| {
            match self.mode {
                CheckpointMode::Min => a.1.best_value.partial_cmp(&b.1.best_value),
                CheckpointMode::Max => b.1.best_value.partial_cmp(&a.1.best_value),
            }
            .unwrap()
        });

        for (path, _) in checkpoints.into_iter().skip(keep_best_n) {
            let base_path = path.with_extension("");
            if let Err(e) = fs::remove_file(&base_path) {
                eprintln!(
                    "Warning: Failed to remove checkpoint file {}: {}",
                    base_path.display(),
                    e
                );
            }
            if let Err(e) = fs::remove_file(&path) {
                eprintln!(
                    "Warning: Failed to remove metadata file {}: {}",
                    path.display(),
                    e
                );
            }
        }

        Ok(())
    }
}

fn save_model_weights(
    model: &dyn Model,
    path: &Path,
    save_format: SaveFormat,
) -> Result<(), BellandeError> {
    // Get state dict directly (it's not a Result)
    let state_dict = model.state_dict();
    let serializable_state: HashMap<String, Vec<f32>> =
        state_dict.into_iter().map(|(k, v)| (k, v.data)).collect();

    match save_format {
        SaveFormat::Json => {
            let file = File::create(path).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to create weights file: {}", e))
            })?;
            serde_json::to_writer(file, &serializable_state).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to serialize weights: {}", e))
            })?;
        }
        SaveFormat::Binary => {
            let file = File::create(path).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to create weights file: {}", e))
            })?;
            bincode::serialize_into(file, &serializable_state).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to serialize weights: {}", e))
            })?;
        }
    }
    Ok(())
}

fn save_model_state(
    model: &dyn Model,
    path: &Path,
    save_format: SaveFormat,
) -> Result<(), BellandeError> {
    // Get state dict directly
    let state_dict = model.state_dict();
    let serializable_state: HashMap<String, Vec<f32>> =
        state_dict.into_iter().map(|(k, v)| (k, v.data)).collect();

    match save_format {
        SaveFormat::Json => {
            let file = File::create(path).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to create model file: {}", e))
            })?;
            serde_json::to_writer(file, &serializable_state).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to serialize model: {}", e))
            })?;
        }
        SaveFormat::Binary => {
            let file = File::create(path).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to create model file: {}", e))
            })?;
            bincode::serialize_into(file, &serializable_state).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to serialize model: {}", e))
            })?;
        }
    }
    Ok(())
}

fn load_weights_inner(
    model: &mut dyn Model,
    path: &Path,
    save_format: SaveFormat,
) -> Result<(), BellandeError> {
    match save_format {
        SaveFormat::Json => {
            let file = File::open(path).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to open weights file: {}", e))
            })?;
            let state_vec: HashMap<String, Vec<f32>> =
                serde_json::from_reader(file).map_err(|e| {
                    BellandeError::RuntimeError(format!("Failed to deserialize weights: {}", e))
                })?;

            let state_dict: HashMap<String, Tensor> = state_vec
                .into_iter()
                .map(|(k, v)| {
                    let len = v.len();
                    (
                        k,
                        Tensor {
                            shape: vec![len],
                            data: v,
                            requires_grad: false,
                            grad: None,
                            grad_fn: None,
                            device: Device::CPU,
                            dtype: DataType::Float32,
                        },
                    )
                })
                .collect();

            model.load_state_dict(state_dict)?;
            Ok(())
        }
        SaveFormat::Binary => {
            let file = File::open(path).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to open weights file: {}", e))
            })?;
            let state_vec: HashMap<String, Vec<f32>> =
                bincode::deserialize_from(file).map_err(|e| {
                    BellandeError::RuntimeError(format!("Failed to deserialize weights: {}", e))
                })?;

            let state_dict: HashMap<String, Tensor> = state_vec
                .into_iter()
                .map(|(k, v)| {
                    let len = v.len();
                    (
                        k,
                        Tensor {
                            shape: vec![len],
                            data: v,
                            requires_grad: false,
                            grad: None,
                            grad_fn: None,
                            device: Device::CPU,
                            dtype: DataType::Float32,
                        },
                    )
                })
                .collect();

            model.load_state_dict(state_dict)?;
            Ok(())
        }
    }
}

fn load_model_inner(
    model: &mut dyn Model,
    path: &Path,
    save_format: SaveFormat,
) -> Result<(), BellandeError> {
    match save_format {
        SaveFormat::Json => {
            let file = File::open(path).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to open model file: {}", e))
            })?;
            let state_vec: HashMap<String, Vec<f32>> =
                serde_json::from_reader(file).map_err(|e| {
                    BellandeError::RuntimeError(format!("Failed to deserialize model: {}", e))
                })?;

            let state_dict: HashMap<String, Tensor> = state_vec
                .into_iter()
                .map(|(k, v)| {
                    let len = v.len();
                    (
                        k,
                        Tensor {
                            shape: vec![len],
                            data: v,
                            requires_grad: false,
                            grad: None,
                            grad_fn: None,
                            device: Device::CPU,
                            dtype: DataType::Float32,
                        },
                    )
                })
                .collect();

            model.load_state_dict(state_dict)?;
            Ok(())
        }
        SaveFormat::Binary => {
            let file = File::open(path).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to open model file: {}", e))
            })?;
            let state_vec: HashMap<String, Vec<f32>> =
                bincode::deserialize_from(file).map_err(|e| {
                    BellandeError::RuntimeError(format!("Failed to deserialize model: {}", e))
                })?;

            let state_dict: HashMap<String, Tensor> = state_vec
                .into_iter()
                .map(|(k, v)| {
                    let len = v.len();
                    (
                        k,
                        Tensor {
                            shape: vec![len],
                            data: v,
                            requires_grad: false,
                            grad: None,
                            grad_fn: None,
                            device: Device::CPU,
                            dtype: DataType::Float32,
                        },
                    )
                })
                .collect();

            model.load_state_dict(state_dict)?;
            Ok(())
        }
    }
}

impl Callback for ModelCheckpoint {
    fn on_epoch_end(
        &mut self,
        epoch: usize,
        logs: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        if let Some(&current) = logs.get(&self.monitor) {
            if !self.save_best_only || self.is_better(current) {
                self.best_value = current;

                let filepath = PathBuf::from(
                    self.filepath
                        .replace("{epoch}", &epoch.to_string())
                        .replace("{val}", &format!("{:.4}", current)),
                );

                self.save_checkpoint(&filepath, epoch, logs)?;
            }
        }
        Ok(())
    }

    fn on_train_begin(&mut self, _logs: &HashMap<String, f32>) -> Result<(), BellandeError> {
        if let Some(parent) = Path::new(&self.filepath).parent() {
            fs::create_dir_all(parent).map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to create checkpoint directory: {}", e))
            })?;
        }

        let meta_pattern = self.filepath.replace("{epoch}", "*").replace("{val}", "*");
        let meta_pattern = format!("{}.meta.json", meta_pattern);

        let existing_checkpoints: Vec<_> = glob::glob(&meta_pattern)
            .map_err(|e| {
                BellandeError::RuntimeError(format!("Failed to read checkpoint directory: {}", e))
            })?
            .filter_map(Result::ok)
            .collect();

        if !existing_checkpoints.is_empty() {
            let mut best_checkpoint = None;
            let mut best_value = match self.mode {
                CheckpointMode::Min => f32::INFINITY,
                CheckpointMode::Max => f32::NEG_INFINITY,
            };

            for checkpoint_path in existing_checkpoints {
                if let Ok(file) = File::open(&checkpoint_path) {
                    if let Ok(metadata) = serde_json::from_reader::<_, CheckpointMetadata>(file) {
                        if self.is_better(metadata.best_value) {
                            best_value = metadata.best_value;
                            best_checkpoint = Some((checkpoint_path, metadata));
                        }
                    }
                }
            }

            if let Some((path, metadata)) = best_checkpoint {
                self.best_value = metadata.best_value;

                if self.verbose {
                    println!(
                        "Resuming from checkpoint: {} (best {} = {})",
                        path.display(),
                        self.monitor,
                        self.best_value
                    );
                }

                // Save format and weights_only flag before borrowing model
                let save_format = self.save_format;
                let save_weights_only = self.save_weights_only;

                if let Some(model) = self.model.as_mut() {
                    let model_path = path.with_extension(match save_format {
                        SaveFormat::Json => "json",
                        SaveFormat::Binary => "bin",
                    });

                    if model_path.exists() {
                        if save_weights_only {
                            load_weights_inner(model.as_mut(), &model_path, save_format)?;
                        } else {
                            load_model_inner(model.as_mut(), &model_path, save_format)?;
                        }
                    }
                }
            }
        } else if self.verbose {
            println!("No existing checkpoints found, starting from scratch");
        }

        Ok(())
    }

    fn on_train_end(&mut self, logs: &HashMap<String, f32>) -> Result<(), BellandeError> {
        if let Some(&final_value) = logs.get(&self.monitor) {
            let filepath = PathBuf::from(
                self.filepath
                    .replace("{epoch}", "final")
                    .replace("{val}", &format!("{:.4}", final_value)),
            );

            let metadata = CheckpointMetadata {
                epoch: usize::MAX,
                best_value: self.best_value,
                monitor: self.monitor.clone(),
                mode: self.mode,
                metrics: logs.clone(),
            };

            // Save format and weights_only flag before borrowing model
            let save_format = self.save_format;
            let save_weights_only = self.save_weights_only;
            let verbose = self.verbose;

            if let Some(model) = self.model.as_ref() {
                if save_weights_only {
                    save_model_weights(model.as_ref(), &filepath, save_format)?;
                } else {
                    save_model_state(model.as_ref(), &filepath, save_format)?;
                }

                let metadata_path = filepath.with_extension("meta.json");
                let file = File::create(metadata_path).map_err(|e| {
                    BellandeError::RuntimeError(format!(
                        "Failed to create final metadata file: {}",
                        e
                    ))
                })?;

                serde_json::to_writer_pretty(file, &metadata).map_err(|e| {
                    BellandeError::RuntimeError(format!("Failed to write final metadata: {}", e))
                })?;

                if verbose {
                    println!(
                        "Saved final checkpoint to {} (best {} = {})",
                        filepath.display(),
                        self.monitor,
                        self.best_value
                    );
                }

                // Clean up old checkpoints if configured
                if let Some(keep_best_n) = self.keep_best_n {
                    self.cleanup_old_checkpoints(keep_best_n)?;
                }
            }
        }

        Ok(())
    }
}
