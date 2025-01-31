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

use crate::core::error::BellandeError;
use crate::core::tensor::Tensor;
use std::fmt;
use std::sync::Arc;

// Make AutogradFunction require Debug
pub trait AutogradFunction: Send + Sync + fmt::Debug {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, BellandeError>;
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, BellandeError>;
}

pub struct AutogradContext {
    saved_tensors: Vec<Tensor>,
    needs_input_grad: Vec<bool>,
}

impl AutogradContext {
    pub fn new(needs_input_grad: Vec<bool>) -> Self {
        AutogradContext {
            saved_tensors: Vec::new(),
            needs_input_grad,
        }
    }

    pub fn save_for_backward(&mut self, tensor: Tensor) {
        self.saved_tensors.push(tensor);
    }

    pub fn get_saved_tensors(&self) -> &[Tensor] {
        &self.saved_tensors
    }
}

// Add Debug implementations for functions
#[derive(Debug)]
pub struct AddFunction;

#[derive(Debug)]
pub struct MulFunction;

#[derive(Debug)]
pub struct MatMulFunction;

impl AutogradFunction for AddFunction {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, BellandeError> {
        if inputs.len() != 2 {
            return Err(BellandeError::InvalidInputs(
                "Add operation requires exactly 2 input tensors".to_string(),
            ));
        }

        let a = inputs[0];
        let b = inputs[1];
        if a.shape != b.shape {
            return Err(BellandeError::DimensionMismatch);
        }

        let mut result_data = Vec::with_capacity(a.data.len());
        for i in 0..a.data.len() {
            result_data.push(a.data[i] + b.data[i]);
        }

        Ok(Tensor {
            data: result_data,
            shape: a.shape.clone(),
            requires_grad: a.requires_grad || b.requires_grad,
            grad: None,
            grad_fn: Some(Arc::new(AddFunction)),
            device: a.device.clone(),
            dtype: a.dtype,
        })
    }

    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Tensor>, BellandeError> {
        Ok(vec![grad_output.clone(), grad_output.clone()])
    }
}
