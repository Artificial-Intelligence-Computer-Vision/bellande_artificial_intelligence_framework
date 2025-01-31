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

pub struct Conv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    pub(crate) weight: Tensor,
    pub(crate) bias: Option<Tensor>,
    pub(crate) input: Option<Tensor>, // Changed from input_cache to input
    pub(crate) weight_grad: Option<Tensor>,
    pub(crate) bias_grad: Option<Tensor>,
    pub(crate) training: bool,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: bool,
    ) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        let padding = padding.unwrap_or((0, 0));
        let weight = Tensor::randn(&[out_channels, in_channels, kernel_size.0, kernel_size.1]);
        let bias = if bias {
            Some(Tensor::zeros(&[out_channels]))
        } else {
            None
        };

        Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias,
            input: None,
            weight_grad: None,
            bias_grad: None,
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

        if channels != self.in_channels {
            return Err(BellandeError::DimensionMismatch);
        }

        // Safe output dimension calculation
        let output_height = ((height as i64 + 2 * self.padding.0 as i64
            - self.kernel_size.0 as i64)
            / self.stride.0 as i64
            + 1) as usize;
        let output_width = ((width as i64 + 2 * self.padding.1 as i64 - self.kernel_size.1 as i64)
            / self.stride.1 as i64
            + 1) as usize;

        // Validate output dimensions
        if output_height == 0 || output_width == 0 {
            return Err(BellandeError::InvalidShape(
                "Convolution resulted in zero output dimensions".into(),
            ));
        }

        let mut output = vec![0.0; batch_size * self.out_channels * output_height * output_width];

        // Implement convolution operation with bounds checking
        for b in 0..batch_size {
            for out_c in 0..self.out_channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let mut sum = 0.0;

                        for in_c in 0..self.in_channels {
                            for k_h in 0..self.kernel_size.0 {
                                for k_w in 0..self.kernel_size.1 {
                                    // Safe input position calculation with padding
                                    let in_h = out_h
                                        .checked_mul(self.stride.0)
                                        .and_then(|h| h.checked_add(k_h))
                                        .and_then(|h| h.checked_sub(self.padding.0));

                                    let in_w = out_w
                                        .checked_mul(self.stride.1)
                                        .and_then(|w| w.checked_add(k_w))
                                        .and_then(|w| w.checked_sub(self.padding.1));

                                    // Check if the input position is valid
                                    if let (Some(h), Some(w)) = (in_h, in_w) {
                                        if h < height && w < width {
                                            let input_idx =
                                                ((b * channels + in_c) * height + h) * width + w;
                                            let weight_idx = ((out_c * self.in_channels + in_c)
                                                * self.kernel_size.0
                                                + k_h)
                                                * self.kernel_size.1
                                                + k_w;

                                            if input_idx < input.data.len()
                                                && weight_idx < self.weight.data.len()
                                            {
                                                sum += input.data[input_idx]
                                                    * self.weight.data[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(ref bias) = self.bias {
                            if out_c < bias.data.len() {
                                sum += bias.data[out_c];
                            }
                        }

                        let output_idx = ((b * self.out_channels + out_c) * output_height + out_h)
                            * output_width
                            + out_w;
                        if output_idx < output.len() {
                            output[output_idx] = sum;
                        }
                    }
                }
            }
        }

        Ok(Tensor::new(
            output,
            vec![batch_size, self.out_channels, output_height, output_width],
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
        let (batch_size, _, height, width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );

        let mut grad_input = vec![0.0; input.data.len()];
        let (_, _, output_height, output_width) = (
            grad_output.shape[0],
            grad_output.shape[1],
            grad_output.shape[2],
            grad_output.shape[3],
        );

        // Compute input gradients
        for b in 0..batch_size {
            for out_c in 0..self.out_channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let out_idx = ((b * self.out_channels + out_c) * output_height + out_h)
                            * output_width
                            + out_w;
                        let grad = grad_output.data[out_idx];

                        for in_c in 0..self.in_channels {
                            for k_h in 0..self.kernel_size.0 {
                                for k_w in 0..self.kernel_size.1 {
                                    let in_h = out_h * self.stride.0 + k_h - self.padding.0;
                                    let in_w = out_w * self.stride.1 + k_w - self.padding.1;

                                    if in_h < height && in_w < width {
                                        let input_idx =
                                            ((b * self.in_channels + in_c) * height + in_h) * width
                                                + in_w;
                                        let weight_idx = ((out_c * self.in_channels + in_c)
                                            * self.kernel_size.0
                                            + k_h)
                                            * self.kernel_size.1
                                            + k_w;
                                        grad_input[input_idx] +=
                                            grad * self.weight.data[weight_idx];
                                    }
                                }
                            }
                        }
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

    pub fn backward_weight(
        &self,
        grad_output: &Tensor,
        input: &Tensor,
    ) -> Result<Tensor, BellandeError> {
        let mut grad_weight = vec![0.0; self.weight.data.len()];
        let (batch_size, _, output_height, output_width) = (
            grad_output.shape[0],
            grad_output.shape[1],
            grad_output.shape[2],
            grad_output.shape[3],
        );

        let (_, _, height, width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );

        // Compute weight gradients
        for b in 0..batch_size {
            for out_c in 0..self.out_channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let out_idx = ((b * self.out_channels + out_c) * output_height + out_h)
                            * output_width
                            + out_w;
                        let grad = grad_output.data[out_idx];

                        for in_c in 0..self.in_channels {
                            for k_h in 0..self.kernel_size.0 {
                                for k_w in 0..self.kernel_size.1 {
                                    let in_h = out_h * self.stride.0 + k_h - self.padding.0;
                                    let in_w = out_w * self.stride.1 + k_w - self.padding.1;

                                    if in_h < height && in_w < width {
                                        let input_idx =
                                            ((b * self.in_channels + in_c) * height + in_h) * width
                                                + in_w;
                                        let weight_idx = ((out_c * self.in_channels + in_c)
                                            * self.kernel_size.0
                                            + k_h)
                                            * self.kernel_size.1
                                            + k_w;
                                        grad_weight[weight_idx] += grad * input.data[input_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Tensor::new(
            grad_weight,
            self.weight.shape.clone(),
            true,
            self.weight.device.clone(),
            self.weight.dtype,
        ))
    }

    fn backward_bias(&self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        if self.bias.is_none() {
            return Err(BellandeError::InvalidParameter("No bias present".into()));
        }

        let mut grad_bias = vec![0.0; self.out_channels];
        let (batch_size, _, output_height, output_width) = (
            grad_output.shape[0],
            grad_output.shape[1],
            grad_output.shape[2],
            grad_output.shape[3],
        );

        // Compute bias gradients
        for b in 0..batch_size {
            for out_c in 0..self.out_channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let out_idx = ((b * self.out_channels + out_c) * output_height + out_h)
                            * output_width
                            + out_w;
                        grad_bias[out_c] += grad_output.data[out_idx];
                    }
                }
            }
        }

        Ok(Tensor::new(
            grad_bias,
            vec![self.out_channels],
            true,
            self.weight.device.clone(),
            self.weight.dtype,
        ))
    }
}

impl NeuralLayer for Conv2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        let output = self.forward_impl(input)?;
        self.input = Some(input.clone());
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        let input = self.input.as_ref().ok_or(BellandeError::InvalidBackward(
            "Forward pass not called before backward".into(),
        ))?;

        let grad_input = self.backward_input(grad_output, input)?;
        let grad_weight = self.backward_weight(grad_output, input)?;
        let grad_bias = if self.bias.is_some() {
            Some(self.backward_bias(grad_output)?)
        } else {
            None
        };

        // Store gradients
        self.weight_grad = Some(grad_weight);
        self.bias_grad = grad_bias;

        Ok(grad_input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref bias) = self.bias {
            params.push(("bias".to_string(), bias.clone()));
        }
        params
    }

    fn set_parameter(&mut self, name: &str, value: Tensor) -> Result<(), BellandeError> {
        match name {
            "weight" => {
                if value.shape == self.weight.shape {
                    self.weight = value;
                    Ok(())
                } else {
                    Err(BellandeError::ShapeMismatch(
                        "Weight shape mismatch".to_string(),
                    ))
                }
            }
            "bias" => {
                if let Some(ref bias) = self.bias {
                    if value.shape == bias.shape {
                        self.bias = Some(value);
                        Ok(())
                    } else {
                        Err(BellandeError::ShapeMismatch(
                            "Bias shape mismatch".to_string(),
                        ))
                    }
                } else {
                    Err(BellandeError::InvalidParameter(
                        "Layer does not use bias".to_string(),
                    ))
                }
            }
            _ => Err(BellandeError::InvalidParameter(format!(
                "Unknown parameter name: {}",
                name
            ))),
        }
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for Conv2d {}
unsafe impl Sync for Conv2d {}
