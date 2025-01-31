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

pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    indices: Option<Vec<usize>>,
    pub(crate) input: Option<Tensor>,
    training: bool,
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize), stride: Option<(usize, usize)>) -> Self {
        let stride = stride.unwrap_or(kernel_size);

        MaxPool2d {
            kernel_size,
            stride,
            indices: None,
            input: None,
            training: true,
        }
    }

    fn forward_impl(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        if input.shape.len() != 4 {
            return Err(BellandeError::InvalidShape(
                "Expected 4D tensor (batch_size, channels, height, width)".into(),
            ));
        }

        let (batch_size, channels, height, width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );

        let output_height = (height - self.kernel_size.0) / self.stride.0 + 1;
        let output_width = (width - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = vec![0.0; batch_size * channels * output_height * output_width];
        let mut indices = vec![0; batch_size * channels * output_height * output_width];

        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..output_height {
                    for w in 0..output_width {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let in_h = h * self.stride.0 + kh;
                                let in_w = w * self.stride.1 + kw;
                                let idx = ((b * channels + c) * height + in_h) * width + in_w;
                                let val = input.data[idx];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = idx;
                                }
                            }
                        }

                        let out_idx = ((b * channels + c) * output_height + h) * output_width + w;
                        output[out_idx] = max_val;
                        indices[out_idx] = max_idx;
                    }
                }
            }
        }

        self.indices = Some(indices);

        Ok(Tensor::new(
            output,
            vec![batch_size, channels, output_height, output_width],
            input.requires_grad,
            input.device.clone(),
            input.dtype,
        ))
    }

    pub fn backward_input(
        &self,
        grad_output: &Tensor,
        input: &Tensor,
    ) -> Result<Tensor, BellandeError> {
        let indices = self.indices.as_ref().ok_or(BellandeError::InvalidBackward(
            "Forward pass not called before backward".into(),
        ))?;

        let mut grad_input = vec![0.0; input.data.len()];

        for (out_idx, &in_idx) in indices.iter().enumerate() {
            grad_input[in_idx] += grad_output.data[out_idx];
        }

        Ok(Tensor::new(
            grad_input,
            input.shape.clone(),
            true,
            input.device.clone(),
            input.dtype,
        ))
    }
}

impl NeuralLayer for MaxPool2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let output = self.forward_impl(input)?;
        self.input = Some(input.clone());
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        let input = self.input.as_ref().ok_or(BellandeError::InvalidBackward(
            "Forward pass not called before backward".into(),
        ))?;
        self.backward_input(grad_output, input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new() // MaxPool2d has no learnable parameters
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        Vec::new() // MaxPool2d has no learnable parameters
    }

    fn set_parameter(&mut self, _name: &str, _value: Tensor) -> Result<(), BellandeError> {
        Err(BellandeError::InvalidParameter(
            "MaxPool2d has no learnable parameters".to_string(),
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
unsafe impl Send for MaxPool2d {}
unsafe impl Sync for MaxPool2d {}
