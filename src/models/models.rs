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

use crate::core::{device::Device, dtype::DataType, error::BellandeError, tensor::Tensor};
use crate::models::sequential::Sequential;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Base model trait defining common functionality for neural networks
pub trait Model: Send + Sync {
    /// Forward pass through the model
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError>;

    /// Backward pass through the model
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, BellandeError>;

    /// Get model parameters
    fn parameters(&self) -> Vec<Tensor>;

    /// Set model to training mode
    fn train(&mut self);

    /// Set model to evaluation mode
    fn eval(&mut self);

    /// Save model to file
    fn save(&self, path: &str) -> Result<(), BellandeError>;

    /// Load model from file
    fn load(&mut self, path: &str) -> Result<(), BellandeError>;

    /// Get model state dictionary
    fn state_dict(&self) -> HashMap<String, Tensor>;

    /// Load model state dictionary
    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>)
        -> Result<(), BellandeError>;
}

/// Model state for serialization
#[derive(Serialize, Deserialize)]
pub struct ModelState {
    pub model_type: String,
    pub state_dict: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
    pub config: ModelConfig,
}

/// Model configuration
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelConfig {
    pub input_shape: Vec<usize>,
    pub num_classes: usize,
    pub dropout_rate: f32,
    pub hidden_layers: Vec<usize>,
}

impl Model for Sequential {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        if self.layers.is_empty() {
            return Err(BellandeError::InvalidInputs("Model has no layers".into()));
        }

        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer
                .forward(&current)
                .map_err(|e| BellandeError::RuntimeError(format!("Forward pass failed: {}", e)))?;
        }
        Ok(current)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, BellandeError> {
        if self.layers.is_empty() {
            return Err(BellandeError::InvalidInputs("Model has no layers".into()));
        }

        if !self.training {
            return Err(BellandeError::InvalidBackward(
                "Model not in training mode".into(),
            ));
        }

        let mut current_grad = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer
                .backward(&current_grad)
                .map_err(|e| BellandeError::RuntimeError(format!("Backward pass failed: {}", e)))?;
        }
        Ok(current_grad)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn save(&self, path: &str) -> Result<(), BellandeError> {
        let state = ModelState {
            model_type: "Sequential".to_string(),
            state_dict: self
                .state_dict()
                .into_iter()
                .map(|(k, v)| (k, v.data))
                .collect(),
            shapes: self
                .state_dict()
                .into_iter()
                .map(|(k, v)| (k, v.shape))
                .collect(),
            config: ModelConfig {
                input_shape: vec![],
                num_classes: 0,
                dropout_rate: 0.0,
                hidden_layers: vec![],
            },
        };

        let file = std::fs::File::create(path)
            .map_err(|e| BellandeError::IOError(format!("Error: {}", e)))?;
        serde_json::to_writer(file, &state).map_err(|e| {
            BellandeError::SerializationError(format!("Failed to serialize model state: {}", e))
        })
    }

    fn load(&mut self, path: &str) -> Result<(), BellandeError> {
        let file = std::fs::File::open(path)
            .map_err(|e| BellandeError::IOError(format!("Error: {}", e)))?;

        let state: ModelState = serde_json::from_reader(file).map_err(|e| {
            BellandeError::SerializationError(format!("Failed to deserialize model state: {}", e))
        })?;

        let mut state_dict = HashMap::new();
        for (key, data) in state.state_dict {
            let shape = state.shapes.get(&key).ok_or_else(|| {
                BellandeError::RuntimeError(format!("Missing shape for key: {}", key))
            })?;

            state_dict.insert(
                key,
                Tensor::new(data, shape.clone(), true, Device::CPU, DataType::Float32),
            );
        }

        self.load_state_dict(state_dict)
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state_dict = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                state_dict.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        state_dict
    }

    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor>,
    ) -> Result<(), BellandeError> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            for (name, _) in layer.named_parameters() {
                let key = format!("layer_{}.{}", i, name);
                if let Some(param) = state_dict.get(&key) {
                    layer.set_parameter(&name, param.clone()).map_err(|e| {
                        BellandeError::RuntimeError(format!(
                            "Failed to set parameter {}: {}",
                            key, e
                        ))
                    })?;
                } else {
                    return Err(BellandeError::RuntimeError(format!(
                        "Missing parameter: {}",
                        key
                    )));
                }
            }
        }
        Ok(())
    }
}

// Thread safety implementations
unsafe impl Send for Sequential {}
unsafe impl Sync for Sequential {}
