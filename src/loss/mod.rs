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

pub mod bce;
pub mod cross_entropy;
pub mod custom;
pub mod mse;

/// The Loss trait defines the interface for loss functions used in training neural networks.
pub trait Loss: Send + Sync {
    fn forward(&self, output: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError>;
    fn backward(&self, output: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError>;

    /// Optional method to get the name of the loss function
    fn name(&self) -> &str {
        "GenericLoss"
    }

    /// Optional method to get the reduction method used by the loss function
    fn reduction(&self) -> Reduction {
        Reduction::Mean
    }
}

/// Enumeration of possible reduction methods for loss functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}

pub trait StaticLoss: Loss + 'static {}
impl<T: Loss + 'static> StaticLoss for T {}

pub trait LossInit: Loss {
    fn new() -> Self;
    fn new_with_reduction(reduction: Reduction) -> Self;
}

pub trait WeightedLoss: Loss {
    /// Computes the forward pass with sample weights
    fn forward_weighted(
        &self,
        output: &Tensor,
        target: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor, BellandeError>;

    /// Computes the backward pass with sample weights
    fn backward_weighted(
        &self,
        output: &Tensor,
        target: &Tensor,
        weights: &Tensor,
    ) -> Result<Tensor, BellandeError>;
}

pub trait ClassWeightedLoss: Loss {
    fn set_class_weights(&mut self, weights: Tensor) -> Result<(), BellandeError>;
    fn get_class_weights(&self) -> Option<&Tensor>;
}

pub mod utils {
    use super::*;

    /// Validates input shapes for loss computation
    pub fn validate_shapes(output: &Tensor, target: &Tensor) -> Result<(), BellandeError> {
        if output.shape != target.shape {
            return Err(BellandeError::ShapeMismatch(format!(
                "Output shape {:?} doesn't match target shape {:?}",
                output.shape, target.shape
            )));
        }
        Ok(())
    }

    /// Applies reduction method to loss values
    pub fn apply_reduction(loss: &Tensor, reduction: Reduction) -> Result<Tensor, BellandeError> {
        let result = match reduction {
            Reduction::None => Ok(loss.clone()),
            Reduction::Mean => {
                let sum: f32 = loss.data.iter().sum();
                let mean = sum / (loss.data.len() as f32);
                Ok(Tensor::new(
                    vec![mean],
                    vec![1],
                    loss.requires_grad,
                    loss.device.clone(),
                    loss.dtype,
                ))
            }
            Reduction::Sum => {
                let sum: f32 = loss.data.iter().sum();
                Ok(Tensor::new(
                    vec![sum],
                    vec![1],
                    loss.requires_grad,
                    loss.device.clone(),
                    loss.dtype,
                ))
            }
        };
        result
    }

    /// Compute element-wise loss without reduction
    pub fn compute_elementwise_loss(
        output: &Tensor,
        target: &Tensor,
        op: impl Fn(f32, f32) -> f32,
    ) -> Result<Tensor, BellandeError> {
        validate_shapes(output, target)?;

        let loss_data: Vec<f32> = output
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(&o, &t)| op(o, t))
            .collect();

        Ok(Tensor::new(
            loss_data,
            output.shape.clone(),
            output.requires_grad,
            output.device.clone(),
            output.dtype,
        ))
    }

    /// Apply weights to loss values
    pub fn apply_weights(loss: &mut Tensor, weights: &Tensor) -> Result<(), BellandeError> {
        if loss.shape != weights.shape {
            return Err(BellandeError::ShapeMismatch(
                "Weights shape doesn't match loss shape".into(),
            ));
        }

        for (l, &w) in loss.data.iter_mut().zip(weights.data.iter()) {
            *l *= w;
        }
        Ok(())
    }
}
