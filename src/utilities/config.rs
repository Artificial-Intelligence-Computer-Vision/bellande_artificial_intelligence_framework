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

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Configuration {
    // Training configuration
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub optimizer: OptimizerConfig,

    // Model configuration
    pub model: ModelConfig,

    // Data configuration
    pub data: DataConfig,

    // System configuration
    pub system: SystemConfig,

    // Custom parameters
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OptimizerConfig {
    pub name: String,
    pub momentum: Option<f32>,
    pub beta1: Option<f32>,
    pub beta2: Option<f32>,
    pub weight_decay: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub input_shape: Vec<usize>,
    pub num_classes: usize,
    pub hidden_layers: Vec<usize>,
    pub dropout_rate: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DataConfig {
    pub train_path: String,
    pub val_path: Option<String>,
    pub test_path: Option<String>,
    pub augmentation: bool,
    pub normalize: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemConfig {
    pub num_workers: usize,
    pub device: String,
    pub precision: String,
    pub seed: Option<u64>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            name: "adam".to_string(),
            momentum: Some(0.9),
            beta1: Some(0.9),
            beta2: Some(0.999),
            weight_decay: Some(0.0),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            architecture: "mlp".to_string(),
            input_shape: vec![784], // Default for MNIST-like data
            num_classes: 10,
            hidden_layers: vec![512, 256],
            dropout_rate: Some(0.5),
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        DataConfig {
            train_path: "data/train".to_string(),
            val_path: Some("data/val".to_string()),
            test_path: Some("data/test".to_string()),
            augmentation: false,
            normalize: true,
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        SystemConfig {
            num_workers: num_cpus::get(),
            device: "cpu".to_string(),
            precision: "float32".to_string(),
            seed: None,
        }
    }
}

impl Default for Configuration {
    fn default() -> Self {
        Configuration {
            batch_size: 32,
            epochs: 10,
            learning_rate: 0.001,
            optimizer: OptimizerConfig::default(),
            model: ModelConfig::default(),
            data: DataConfig::default(),
            system: SystemConfig::default(),
            parameters: HashMap::new(),
        }
    }
}

impl Configuration {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let content = fs::read_to_string(path)?;
        let config: Configuration = serde_yaml::from_str(&content)?;

        if let Err(validation_error) = config.validate() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                validation_error,
            )));
        }

        Ok(config)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let content = serde_yaml::to_string(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        // Validate batch size
        if self.batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }

        // Validate learning rate
        if self.learning_rate <= 0.0 {
            return Err("Learning rate must be positive".to_string());
        }

        // Validate model configuration
        if self.model.input_shape.is_empty() {
            return Err("Input shape cannot be empty".to_string());
        }

        if self.model.num_classes == 0 {
            return Err("Number of classes must be greater than 0".to_string());
        }

        // Validate optimizer configuration
        if let Some(momentum) = self.optimizer.momentum {
            if !(0.0..=1.0).contains(&momentum) {
                return Err("Momentum must be between 0 and 1".to_string());
            }
        }

        if let Some(beta1) = self.optimizer.beta1 {
            if !(0.0..=1.0).contains(&beta1) {
                return Err("Beta1 must be between 0 and 1".to_string());
            }
        }

        if let Some(beta2) = self.optimizer.beta2 {
            if !(0.0..=1.0).contains(&beta2) {
                return Err("Beta2 must be between 0 and 1".to_string());
            }
        }

        // Validate data paths
        if !Path::new(&self.data.train_path).exists() {
            return Err("Training data path does not exist".to_string());
        }

        if let Some(val_path) = &self.data.val_path {
            if !Path::new(val_path).exists() {
                return Err("Validation data path does not exist".to_string());
            }
        }

        if let Some(test_path) = &self.data.test_path {
            if !Path::new(test_path).exists() {
                return Err("Test data path does not exist".to_string());
            }
        }

        // Validate system configuration
        if self.system.num_workers == 0 {
            return Err("Number of workers must be greater than 0".to_string());
        }

        match self.system.precision.as_str() {
            "float32" | "float16" | "bfloat16" => Ok(()),
            _ => Err("Invalid precision format".to_string()),
        }?;

        Ok(())
    }

    pub fn merge(&mut self, other: &Configuration) {
        // Merge only non-default values from other configuration
        if other.batch_size != Configuration::default().batch_size {
            self.batch_size = other.batch_size;
        }
        if other.epochs != Configuration::default().epochs {
            self.epochs = other.epochs;
        }
        if other.learning_rate != Configuration::default().learning_rate {
            self.learning_rate = other.learning_rate;
        }

        // Merge parameters
        self.parameters.extend(other.parameters.clone());
    }
}
