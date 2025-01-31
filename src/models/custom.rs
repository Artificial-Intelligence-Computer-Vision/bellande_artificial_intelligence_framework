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
use crate::layer::batch_norm::BatchNorm1d;
use crate::layer::dropout::Dropout;
use crate::layer::{activation::ReLU, linear::Linear};
use crate::models::sequential::Sequential;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::Path;

pub trait ModelBuilder {
    fn build(&self, config: &ModelConfig) -> Result<Box<dyn Model>, BellandeError>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub input_shape: Vec<usize>,
    pub num_classes: usize,
    pub hyperparameters: HashMap<String, f32>,
}

pub trait Model {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor, BellandeError>;
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, BellandeError>;
    fn parameters(&self) -> Vec<Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
    fn save(&self, path: &str) -> Result<(), BellandeError>;
    fn load(&mut self, path: &str) -> Result<(), BellandeError>;
}

// Remove derive for Sequential
pub struct CustomModel {
    layers: Sequential,
    config: ModelConfig,
    training: bool,
}

#[derive(Serialize, Deserialize)]
struct ModelState {
    config: ModelConfig,
    parameters: Vec<Vec<f32>>,
    parameter_shapes: Vec<Vec<usize>>,
}

impl CustomModel {
    pub fn new(config: ModelConfig) -> Self {
        let mut layers = Sequential::new();
        let input_size = config.input_shape.iter().product();
        let hidden_size = *config.hyperparameters.get("hidden_size").unwrap_or(&128.0) as usize;

        // Build the model architecture based on config
        layers.add(Box::new(Linear::new(input_size, hidden_size, true)));
        layers.add(Box::new(ReLU::new()));

        if let Some(dropout_rate) = config.hyperparameters.get("dropout_rate") {
            // Handle the Result from Dropout::new
            if let Ok(dropout) = Dropout::new(*dropout_rate) {
                layers.add(Box::new(dropout));
            }
        }

        // Add batch normalization if specified
        if config.hyperparameters.get("use_batch_norm").unwrap_or(&0.0) > &0.0 {
            layers.add(Box::new(BatchNorm1d::new(hidden_size, 1e-5, 0.1, true)));
        }

        // Add additional layers based on depth parameter
        let depth = *config.hyperparameters.get("depth").unwrap_or(&1.0) as usize;
        for _ in 0..depth {
            layers.add(Box::new(Linear::new(hidden_size, hidden_size, true)));
            layers.add(Box::new(ReLU::new()));

            if let Some(dropout_rate) = config.hyperparameters.get("dropout_rate") {
                // Handle the Result from Dropout::new
                if let Ok(dropout) = Dropout::new(*dropout_rate) {
                    layers.add(Box::new(dropout));
                }
            }
        }

        // Output layer
        layers.add(Box::new(Linear::new(hidden_size, config.num_classes, true)));

        CustomModel {
            layers,
            config,
            training: true,
        }
    }

    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config
            .hyperparameters
            .insert("learning_rate".to_string(), lr);
    }

    fn create_checkpoint_dir(&self, path: &str) -> Result<(), BellandeError> {
        if let Some(parent) = Path::new(path).parent() {
            create_dir_all(parent).map_err(|e| {
                BellandeError::IOError(format!("Failed to create directory: {}", e).into())
            })?;
        }
        Ok(())
    }
}

impl Model for CustomModel {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor, BellandeError> {
        if x.shape[1..] != self.config.input_shape[..] {
            return Err(BellandeError::InvalidShape(format!(
                "Expected input shape {:?}, got {:?}",
                self.config.input_shape,
                x.shape[1..].to_vec()
            )));
        }
        self.layers.forward(x)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, BellandeError> {
        if !self.training {
            return Err(BellandeError::InvalidOperation(
                "Backward pass called while model is in evaluation mode".into(),
            ));
        }
        self.layers.backward(grad)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.parameters()
    }

    fn train(&mut self) {
        self.training = true;
        self.layers.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.layers.eval();
    }

    fn save(&self, path: &str) -> Result<(), BellandeError> {
        self.create_checkpoint_dir(path)?;

        let parameters: Vec<Vec<f32>> = self
            .parameters()
            .iter()
            .map(|tensor| tensor.data.clone())
            .collect();

        let parameter_shapes: Vec<Vec<usize>> = self
            .parameters()
            .iter()
            .map(|tensor| tensor.shape.clone())
            .collect();

        let model_state = ModelState {
            config: self.config.clone(),
            parameters,
            parameter_shapes,
        };

        let serialized = serde_json::to_string(&model_state).map_err(|e| {
            BellandeError::SerializationError(format!("Failed to serialize model: {}", e))
        })?;

        let mut file = File::create(path)
            .map_err(|e| BellandeError::IOError(format!("Failed to create file: {}", e)))?;

        file.write_all(serialized.as_bytes())
            .map_err(|e| BellandeError::IOError(format!("Failed to write to file: {}", e)))?;

        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<(), BellandeError> {
        let mut file = File::open(path)
            .map_err(|e| BellandeError::IOError(format!("Failed to open file: {}", e)))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| BellandeError::IOError(format!("Failed to read file: {}", e)))?;

        let model_state: ModelState = serde_json::from_str(&contents).map_err(|e| {
            BellandeError::SerializationError(format!("Failed to deserialize model: {}", e))
        })?;

        // Verify configuration compatibility
        if model_state.config.input_shape != self.config.input_shape {
            return Err(BellandeError::InvalidConfiguration(
                "Input shape mismatch".into(),
            ));
        }

        if model_state.config.num_classes != self.config.num_classes {
            return Err(BellandeError::InvalidConfiguration(
                "Number of classes mismatch".into(),
            ));
        }

        // Load parameters
        let mut current_parameters = self.parameters();
        if current_parameters.len() != model_state.parameters.len() {
            return Err(BellandeError::InvalidConfiguration(
                "Parameter count mismatch".into(),
            ));
        }

        for (((param, saved_data), saved_shape), current_param) in current_parameters
            .iter_mut()
            .zip(model_state.parameters.iter())
            .zip(model_state.parameter_shapes.iter())
            .zip(self.parameters())
        {
            if saved_shape != &current_param.shape {
                return Err(BellandeError::InvalidConfiguration(format!(
                    "Parameter shape mismatch: expected {:?}, got {:?}",
                    current_param.shape, saved_shape
                )));
            }
            param.data = saved_data.clone();
        }

        self.config = model_state.config;
        Ok(())
    }
}
