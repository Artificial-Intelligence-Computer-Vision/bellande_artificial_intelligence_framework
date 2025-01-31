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
use crate::loss::{bce::Reduction, Loss};

pub struct MSELoss {
    reduction: Reduction,
}

impl MSELoss {
    pub fn new(reduction: Reduction) -> Self {
        MSELoss { reduction }
    }
}

// Implement Loss trait for MSELoss
impl Loss for MSELoss {
    fn forward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError> {
        if prediction.shape != target.shape {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut loss = Vec::with_capacity(prediction.data.len());
        for (pred, tgt) in prediction.data.iter().zip(target.data.iter()) {
            loss.push((pred - tgt).powi(2));
        }

        match self.reduction {
            Reduction::None => Ok(Tensor::new(
                loss,
                prediction.shape.clone(),
                true,
                prediction.device.clone(),
                prediction.dtype,
            )),
            Reduction::Mean => Ok(Tensor::new(
                vec![loss.iter().sum::<f32>() / loss.len() as f32],
                vec![1],
                true,
                prediction.device.clone(),
                prediction.dtype,
            )),
            Reduction::Sum => Ok(Tensor::new(
                vec![loss.iter().sum()],
                vec![1],
                true,
                prediction.device.clone(),
                prediction.dtype,
            )),
        }
    }

    fn backward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError> {
        if prediction.shape != target.shape {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut grad = Vec::with_capacity(prediction.data.len());
        for (pred, tgt) in prediction.data.iter().zip(target.data.iter()) {
            // Derivative of (pred - tgt)^2 is 2(pred - tgt)
            grad.push(2.0 * (pred - tgt));
        }

        let grad = match self.reduction {
            Reduction::None => grad,
            Reduction::Mean => {
                // Scale gradients by 1/N for mean reduction
                let scale = 1.0 / prediction.data.len() as f32;
                grad.iter().map(|&g| g * scale).collect()
            }
            Reduction::Sum => grad,
        };

        Ok(Tensor::new(
            grad,
            prediction.shape.clone(),
            true,
            prediction.device.clone(),
            prediction.dtype,
        ))
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for MSELoss {}
unsafe impl Sync for MSELoss {}
