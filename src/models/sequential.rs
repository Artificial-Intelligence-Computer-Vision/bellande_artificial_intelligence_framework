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

/// Trait defining a neural network layer
pub trait NeuralLayer: Send + Sync {
    /// Forward pass
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError>;

    /// Backward pass
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, BellandeError>;

    /// Get layer parameters
    fn parameters(&self) -> Vec<Tensor>;

    /// Get named parameters
    fn named_parameters(&self) -> Vec<(String, Tensor)>;

    /// Set parameter value
    fn set_parameter(&mut self, name: &str, value: Tensor) -> Result<(), BellandeError>;

    /// Set layer to training mode
    fn train(&mut self);

    /// Set layer to evaluation mode
    fn eval(&mut self);
}

/// Sequential container for neural network layers
pub struct Sequential {
    pub(crate) layers: Vec<Box<dyn NeuralLayer>>,
    pub(crate) training: bool,
}

impl Sequential {
    /// Creates a new empty Sequential container
    pub fn new() -> Self {
        Sequential {
            layers: Vec::new(),
            training: true,
        }
    }

    /// Adds a layer to the container and returns mutable reference for chaining
    pub fn add(&mut self, layer: Box<dyn NeuralLayer>) -> &mut Self {
        self.layers.push(layer);
        self
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    /// Backward pass through all layers in reverse order
    pub fn backward(&mut self, grad: &Tensor) -> Result<Tensor, BellandeError> {
        if !self.training {
            return Err(BellandeError::InvalidBackward(
                "Forward pass not called before backward".into(),
            ))?;
        }

        let mut current_grad = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad)?;
        }
        Ok(current_grad)
    }

    /// Get all parameters from all layers
    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    /// Get number of layers
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if container is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get layer at index
    pub fn get_layer(&self, index: usize) -> Option<&Box<dyn NeuralLayer>> {
        self.layers.get(index)
    }

    /// Get mutable layer at index
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut Box<dyn NeuralLayer>> {
        self.layers.get_mut(index)
    }

    /// Set model to training mode
    pub fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    /// Set model to evaluation mode
    pub fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }
}

// Implement Default for Sequential
impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}
