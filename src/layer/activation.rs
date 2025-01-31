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

pub trait Activation {
    fn forward(&self, input: &Tensor) -> Result<Tensor, BellandeError>;
    fn backward(&self, grad_output: &Tensor) -> Result<Tensor, BellandeError>;
}

pub struct ReLU {
    mask: Option<Vec<bool>>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { mask: None }
    }
}

impl Activation for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let mut output = input.data.clone();
        let mask: Vec<bool> = output
            .iter_mut()
            .map(|x| {
                if *x < 0.0 {
                    *x = 0.0;
                    false
                } else {
                    true
                }
            })
            .collect();

        Ok(Tensor::new(
            output,
            input.shape.clone(),
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }

    fn backward(&self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        if let Some(ref mask) = self.mask {
            let grad = grad_output
                .data
                .iter()
                .zip(mask.iter())
                .map(|(&g, &m)| if m { g } else { 0.0 })
                .collect();

            Ok(Tensor::new(
                grad,
                grad_output.shape.clone(),
                true,
                grad_output.device.clone(),
                grad_output.dtype,
            ))
        } else {
            Err(BellandeError::RuntimeError(
                "Forward pass not called".into(),
            ))
        }
    }
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let output = input
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        Ok(Tensor::new(
            output,
            input.shape.clone(),
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }

    fn backward(&self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        let grad = grad_output
            .data
            .iter()
            .map(|&x| {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            })
            .collect();

        Ok(Tensor::new(
            grad,
            grad_output.shape.clone(),
            true,
            grad_output.device.clone(),
            grad_output.dtype,
        ))
    }
}
