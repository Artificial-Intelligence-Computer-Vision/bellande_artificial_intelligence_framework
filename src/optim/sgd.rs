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
use crate::optim::{Optimizer, OptimizerState, ParameterGroup};
use std::collections::HashMap;

pub struct SGD {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: HashMap<usize, Vec<f32>>,
    param_groups: Vec<ParameterGroup>,
    state: OptimizerState,
}

impl SGD {
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> Self {
        let mut velocity = HashMap::new();
        if momentum > 0.0 {
            for (idx, param) in params.iter().enumerate() {
                velocity.insert(idx, vec![0.0; param.data.len()]);
            }
        }

        // Create initial parameter group with correct types
        let default_group = ParameterGroup::new(params.clone())
            .with_lr(lr)
            .with_weight_decay(weight_decay)
            .with_momentum(momentum)
            .with_eps(1e-8); // Default epsilon value

        SGD {
            params,
            lr,
            momentum,
            weight_decay,
            nesterov,
            velocity,
            param_groups: vec![default_group],
            state: OptimizerState::default(),
        }
    }

    pub fn step(&mut self) -> Result<(), BellandeError> {
        for (idx, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = &param.grad {
                let v_ref = if self.momentum > 0.0 {
                    self.velocity.get_mut(&idx)
                } else {
                    None
                };

                if let Some(v) = v_ref {
                    // Case with momentum
                    for ((p, g), v_i) in param.data.iter_mut().zip(grad.iter()).zip(v.iter_mut()) {
                        let mut d_p = *g;
                        if self.weight_decay != 0.0 {
                            d_p += self.weight_decay * *p;
                        }

                        *v_i = self.momentum * *v_i + d_p;
                        if self.nesterov {
                            d_p += self.momentum * *v_i;
                        } else {
                            d_p = *v_i;
                        }
                        *p -= self.lr * d_p;
                    }
                } else {
                    // Case without momentum
                    for (p, g) in param.data.iter_mut().zip(grad.iter()) {
                        let mut d_p = *g;
                        if self.weight_decay != 0.0 {
                            d_p += self.weight_decay * *p;
                        }
                        *p -= self.lr * d_p;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn zero_grad(&mut self) {
        for param in &mut self.params {
            if let Some(grad) = &mut param.grad {
                grad.iter_mut().for_each(|g| *g = 0.0);
            }
        }
    }
}

// Implement the Optimizer trait
impl Optimizer for SGD {
    fn step(&mut self) -> Result<(), BellandeError> {
        self.step()
    }

    fn zero_grad(&mut self) {
        self.zero_grad()
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn parameters(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn parameters_mut(&mut self) -> &mut Vec<Tensor> {
        &mut self.params
    }

    // Add the missing required methods
    fn get_param_groups(&self) -> &[ParameterGroup] {
        &self.param_groups
    }

    fn get_param_groups_mut(&mut self) -> &mut [ParameterGroup] {
        &mut self.param_groups
    }

    fn add_param_group(&mut self, mut group: ParameterGroup) {
        let start_idx = self.params.len();

        if self.momentum > 0.0 {
            for (i, param) in group.params.iter().enumerate() {
                self.velocity
                    .insert(start_idx + i, vec![0.0; param.data.len()]);
            }
        }

        self.params.extend(group.params.clone());
        self.param_groups.push(group);
    }

    fn state(&self) -> &OptimizerState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut OptimizerState {
        &mut self.state
    }
}

// Implement Send and Sync
unsafe impl Send for SGD {}
unsafe impl Sync for SGD {}
