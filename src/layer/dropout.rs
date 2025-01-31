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
use crate::models::sequential::NeuralLayer;
use rand::Rng;

pub struct Dropout {
    p: f32,
    mask: Option<Vec<bool>>,
    pub(crate) training: bool,
    input: Option<Tensor>,
}

impl Dropout {
    pub fn new(p: f32) -> Result<Self, BellandeError> {
        if !(0.0..1.0).contains(&p) {
            return Err(BellandeError::InvalidParameter(
                "Dropout probability must be between 0 and 1".into(),
            ));
        }

        Ok(Dropout {
            p,
            mask: None,
            training: true,
            input: None,
        })
    }

    fn forward_impl(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        if !self.training {
            return Ok(input.clone());
        }

        let mut rng = rand::thread_rng();
        let mask: Vec<bool> = (0..input.data.len())
            .map(|_| rng.gen::<f32>() > self.p)
            .collect();

        let scale = 1.0 / (1.0 - self.p);
        let output: Vec<f32> = input
            .data
            .iter()
            .zip(mask.iter())
            .map(|(&x, &m)| if m { x * scale } else { 0.0 })
            .collect();

        self.mask = Some(mask);

        Ok(Tensor::new(
            output,
            input.shape.clone(),
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }

    fn backward_input(&self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        let mask = self.mask.as_ref().ok_or_else(|| {
            BellandeError::InvalidBackward("Forward pass not called before backward".into())
        })?;

        let scale = 1.0 / (1.0 - self.p);
        let grad: Vec<f32> = grad_output
            .data
            .iter()
            .zip(mask.iter())
            .map(|(&g, &m)| if m { g * scale } else { 0.0 })
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

impl NeuralLayer for Dropout {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let output = self.forward_impl(input)?;
        self.input = Some(input.clone());
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        let _input = self.input.as_ref().ok_or(BellandeError::InvalidBackward(
            "Forward pass not called before backward".into(),
        ))?;

        self.backward_input(grad_output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new() // Dropout has no learnable parameters
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        Vec::new() // Dropout has no learnable parameters
    }

    fn set_parameter(&mut self, _name: &str, _value: Tensor) -> Result<(), BellandeError> {
        Err(BellandeError::InvalidParameter(
            "Dropout has no learnable parameters".to_string(),
        ))
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for Dropout {}
unsafe impl Sync for Dropout {}
