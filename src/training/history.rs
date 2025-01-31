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

use crate::core::error::BellandeError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    pub epochs: Vec<usize>,
    pub metrics: HashMap<String, Vec<f32>>,
}

impl TrainingHistory {
    pub fn new() -> Self {
        TrainingHistory {
            epochs: Vec::new(),
            metrics: HashMap::new(),
        }
    }

    pub fn update(&mut self, epoch: usize, metrics: HashMap<String, f32>) {
        self.epochs.push(epoch);
        for (key, value) in metrics {
            self.metrics.entry(key).or_insert_with(Vec::new).push(value);
        }
    }

    pub fn get_metric(&self, name: &str) -> Option<&Vec<f32>> {
        self.metrics.get(name)
    }

    pub fn save(&self, path: &str) -> Result<(), BellandeError> {
        let json = serde_json::to_string(self)
            .map_err(|e| BellandeError::RuntimeError(format!("Serialization failed: {}", e)))?;

        fs::write(path, json).map_err(|e| BellandeError::IOError(format!("Error: {}", e)))?;

        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, BellandeError> {
        let json = fs::read_to_string(path)
            .map_err(|e| BellandeError::IOError(format!("Error: {}", e)))?;

        serde_json::from_str(&json)
            .map_err(|e| BellandeError::RuntimeError(format!("Deserialization failed: {}", e)))
    }
}
