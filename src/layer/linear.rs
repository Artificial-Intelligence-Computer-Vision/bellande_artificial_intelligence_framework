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

pub struct Linear {
    in_features: usize,
    out_features: usize,
    pub(crate) weight: Tensor,
    pub(crate) bias: Option<Tensor>,
    input_cache: Option<Tensor>,
    pub(crate) weight_grad: Option<Tensor>,
    pub(crate) bias_grad: Option<Tensor>,
    pub(crate) training: bool,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let weight = Tensor::randn(&[out_features, in_features]);
        let bias = if bias {
            Some(Tensor::zeros(&[out_features]))
        } else {
            None
        };

        Linear {
            in_features,
            out_features,
            weight,
            bias,
            input_cache: None,
            weight_grad: None,
            bias_grad: None,
            training: true,
        }
    }

    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        if input.shape.len() != 2 {
            return Err(BellandeError::InvalidShape(format!("Linear Invalid")))?;
        }

        let batch_size = input.shape[0];
        if input.shape[1] != self.in_features {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut output = vec![0.0; batch_size * self.out_features];

        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut sum = 0.0;
                for i in 0..self.in_features {
                    sum += input.data[b * self.in_features + i]
                        * self.weight.data[o * self.in_features + i];
                }
                if let Some(ref bias) = self.bias {
                    sum += bias.data[o];
                }
                output[b * self.out_features + o] = sum;
            }
        }

        self.input_cache = Some(input.clone());

        Ok(Tensor::new(
            output,
            vec![batch_size, self.out_features],
            true,
            input.device.clone(),
            input.dtype,
        ))
    }
}

impl NeuralLayer for Linear {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, BellandeError> {
        self.forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor, BellandeError> {
        let (grad_input, grad_weight, grad_bias) = self.compute_gradients(grad_output)?;

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

impl Linear {
    fn compute_gradients(
        &self,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor, Option<Tensor>), BellandeError> {
        if let Some(ref input) = self.input_cache {
            let batch_size = grad_output.shape[0];

            // Gradient with respect to input
            let mut grad_input = vec![0.0; input.data.len()];
            // Gradient with respect to weight
            let mut grad_weight = vec![0.0; self.weight.data.len()];
            // Gradient with respect to bias
            let mut grad_bias = if self.bias.is_some() {
                Some(vec![0.0; self.out_features])
            } else {
                None
            };

            // Compute gradients
            for b in 0..batch_size {
                for o in 0..self.out_features {
                    for i in 0..self.in_features {
                        let grad = grad_output.data[b * self.out_features + o];
                        grad_input[b * self.in_features + i] +=
                            grad * self.weight.data[o * self.in_features + i];
                        grad_weight[o * self.in_features + i] +=
                            grad * input.data[b * self.in_features + i];
                    }
                    if let Some(ref mut bias) = grad_bias {
                        bias[o] += grad_output.data[b * self.out_features + o];
                    }
                }
            }

            Ok((
                Tensor::new(
                    grad_input,
                    input.shape.clone(),
                    true,
                    input.device.clone(),
                    input.dtype,
                ),
                Tensor::new(
                    grad_weight,
                    self.weight.shape.clone(),
                    true,
                    self.weight.device.clone(),
                    self.weight.dtype,
                ),
                grad_bias.map(|bias| {
                    Tensor::new(
                        bias,
                        vec![self.out_features],
                        true,
                        self.weight.device.clone(),
                        self.weight.dtype,
                    )
                }),
            ))
        } else {
            Err(BellandeError::RuntimeError(
                "Forward pass not called".into(),
            ))
        }
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for Linear {}
unsafe impl Sync for Linear {}
