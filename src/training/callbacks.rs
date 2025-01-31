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

use crate::core::{error::BellandeError, tensor::Tensor};
use std::collections::HashMap;

pub trait Callback: Send + Sync {
    fn on_epoch_begin(
        &mut self,
        epoch: usize,
        logs: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        Ok(())
    }
    fn on_epoch_end(
        &mut self,
        epoch: usize,
        logs: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        Ok(())
    }
    fn on_batch_begin(
        &mut self,
        batch: usize,
        logs: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        Ok(())
    }
    fn on_batch_end(
        &mut self,
        batch: usize,
        logs: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        Ok(())
    }
    fn on_train_begin(&mut self, logs: &HashMap<String, f32>) -> Result<(), BellandeError> {
        Ok(())
    }
    fn on_train_end(&mut self, logs: &HashMap<String, f32>) -> Result<(), BellandeError> {
        Ok(())
    }
}

pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    monitor: String,
    best_value: f32,
    wait: usize,
    stopped_epoch: usize,
    restore_best_weights: bool,
    best_weights: Option<Vec<Tensor>>,
}

impl EarlyStopping {
    pub fn new(
        patience: usize,
        min_delta: f32,
        monitor: String,
        restore_best_weights: bool,
    ) -> Self {
        EarlyStopping {
            patience,
            min_delta,
            monitor,
            best_value: f32::INFINITY,
            wait: 0,
            stopped_epoch: 0,
            restore_best_weights,
            best_weights: None,
        }
    }
}

impl Callback for EarlyStopping {
    fn on_epoch_end(
        &mut self,
        epoch: usize,
        logs: &HashMap<String, f32>,
    ) -> Result<(), BellandeError> {
        if let Some(&current) = logs.get(&self.monitor) {
            if current < self.best_value - self.min_delta {
                self.best_value = current;
                self.wait = 0;
                if self.restore_best_weights {
                    // Save current weights
                }
            } else {
                self.wait += 1;
                if self.wait >= self.patience {
                    self.stopped_epoch = epoch;
                    return Err(BellandeError::EarlyStopping(format!(
                        "Stopped at epoch {}",
                        epoch
                    )));
                }
            }
        }
        Ok(())
    }
}
