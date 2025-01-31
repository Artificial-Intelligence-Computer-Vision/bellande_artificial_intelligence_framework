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

pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    pub(crate) input: Option<Tensor>,
    training: bool,
}

impl AvgPool2d {
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        let padding = padding.unwrap_or((0, 0));

        AvgPool2d {
            kernel_size,
            stride,
            padding,
            input: None,
            training: true,
        }
    }

    // Internal forward implementation
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

        let output_height = (height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let output_width = (width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = vec![0.0; batch_size * channels * output_height * output_width];

        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..output_height {
                    for ow in 0..output_width {
                        let mut sum = 0.0;
                        let mut count = 0.0;

                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let h = oh as isize * self.stride.0 as isize + kh as isize
                                    - self.padding.0 as isize;
                                let w = ow as isize * self.stride.1 as isize + kw as isize
                                    - self.padding.1 as isize;

                                if h >= 0 && h < height as isize && w >= 0 && w < width as isize {
                                    let input_idx = ((b * channels + c) * height + h as usize)
                                        * width
                                        + w as usize;
                                    sum += input.data[input_idx];
                                    count += 1.0;
                                }
                            }
                        }

                        let output_idx =
                            ((b * channels + c) * output_height + oh) * output_width + ow;
                        output[output_idx] = if count > 0.0 { sum / count } else { 0.0 };
                    }
                }
            }
        }

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
        let (batch_size, channels, height, width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );

        let mut grad_input = vec![0.0; input.data.len()];
        let kernel_size = (self.kernel_size.0 * self.kernel_size.1) as f32;

        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let mut grad = 0.0;

                        let oh_start = (h.saturating_sub(self.kernel_size.0 - 1) + self.stride.0
                            - 1)
                            / self.stride.0;
                        let ow_start = (w.saturating_sub(self.kernel_size.1 - 1) + self.stride.1
                            - 1)
                            / self.stride.1;

                        let oh_end = (h + self.padding.0) / self.stride.0;
                        let ow_end = (w + self.padding.1) / self.stride.1;

                        for oh in oh_start..=oh_end {
                            for ow in ow_start..=ow_end {
                                if oh < grad_output.shape[2] && ow < grad_output.shape[3] {
                                    let output_idx = ((b * channels + c) * grad_output.shape[2]
                                        + oh)
                                        * grad_output.shape[3]
                                        + ow;
                                    grad += grad_output.data[output_idx] / kernel_size;
                                }
                            }
                        }

                        let input_idx = ((b * channels + c) * height + h) * width + w;
                        grad_input[input_idx] = grad;
                    }
                }
            }
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

impl NeuralLayer for AvgPool2d {
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
        Vec::new() // AvgPool2d has no learnable parameters
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        Vec::new() // AvgPool2d has no learnable parameters
    }

    fn set_parameter(&mut self, _name: &str, _value: Tensor) -> Result<(), BellandeError> {
        Err(BellandeError::InvalidParameter(
            "AvgPool2d has no learnable parameters".to_string(),
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
unsafe impl Send for AvgPool2d {}
unsafe impl Sync for AvgPool2d {}
