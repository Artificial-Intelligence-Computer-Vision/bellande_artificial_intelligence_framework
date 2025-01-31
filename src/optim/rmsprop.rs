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

pub struct RMSprop {
    params: Vec<Tensor>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    v: HashMap<usize, Vec<f32>>,   // Square average
    g: HashMap<usize, Vec<f32>>,   // Gradient average (if centered)
    buf: HashMap<usize, Vec<f32>>, // Momentum buffer
    param_groups: Vec<ParameterGroup>,
    state: OptimizerState,
}

impl RMSprop {
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        alpha: f32,
        eps: f32,
        weight_decay: f32,
        momentum: f32,
        centered: bool,
    ) -> Self {
        let mut v = HashMap::new();
        let mut g = HashMap::new();
        let mut buf = HashMap::new();

        for (idx, param) in params.iter().enumerate() {
            v.insert(idx, vec![0.0; param.data.len()]);
            if centered {
                g.insert(idx, vec![0.0; param.data.len()]);
            }
            if momentum > 0.0 {
                buf.insert(idx, vec![0.0; param.data.len()]);
            }
        }

        // Create default parameter group
        let default_group = ParameterGroup::new(params.clone())
            .with_lr(lr)
            .with_weight_decay(weight_decay)
            .with_momentum(momentum)
            .with_eps(eps);

        RMSprop {
            params,
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            v,
            g,
            buf,
            param_groups: vec![default_group],
            state: OptimizerState::new(),
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) -> Result<(), BellandeError> {
        self.state.increment_step();

        for (idx, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = &param.grad {
                let v = self.v.get_mut(&idx).unwrap();
                let mut g = if self.centered {
                    Some(self.g.get_mut(&idx).unwrap())
                } else {
                    None
                };
                let mut buf = if self.momentum > 0.0 {
                    Some(self.buf.get_mut(&idx).unwrap())
                } else {
                    None
                };

                // Process all elements for this parameter
                for i in 0..param.data.len() {
                    let grad_val = grad[i];
                    let mut final_grad = grad_val;

                    // Apply weight decay if needed
                    if self.weight_decay != 0.0 {
                        final_grad += self.weight_decay * param.data[i];
                    }

                    // Update running average of squared gradients
                    v[i] = self.alpha * v[i] + (1.0 - self.alpha) * final_grad * final_grad;

                    if let Some(g_avg) = &mut g {
                        // Update gradient average for centered variant
                        g_avg[i] = self.alpha * g_avg[i] + (1.0 - self.alpha) * final_grad;
                        let denom = (v[i].sqrt() - g_avg[i].powi(2) + self.eps).sqrt();
                        final_grad /= denom;
                    } else {
                        final_grad /= (v[i] + self.eps).sqrt();
                    }

                    if let Some(buf_val) = &mut buf {
                        // Apply momentum if enabled
                        buf_val[i] = self.momentum * buf_val[i] + final_grad;
                        param.data[i] -= self.lr * buf_val[i];
                    } else {
                        param.data[i] -= self.lr * final_grad;
                    }
                }
            }
        }
        Ok(())
    }
    fn zero_grad(&mut self) {
        for param in &mut self.params {
            if let Some(grad) = &mut param.grad {
                grad.iter_mut().for_each(|g| *g = 0.0);
            }
        }
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn parameters(&self) -> &Vec<Tensor> {
        &self.params
    }

    fn parameters_mut(&mut self) -> &mut Vec<Tensor> {
        &mut self.params
    }

    fn name(&self) -> &str {
        "RMSprop"
    }

    fn get_param_groups(&self) -> &[ParameterGroup] {
        &self.param_groups
    }

    fn get_param_groups_mut(&mut self) -> &mut [ParameterGroup] {
        &mut self.param_groups
    }

    fn add_param_group(&mut self, mut group: ParameterGroup) {
        let start_idx = self.params.len();

        // Initialize state for new parameters
        for (i, param) in group.params.iter().enumerate() {
            self.v.insert(start_idx + i, vec![0.0; param.data.len()]);
            if self.centered {
                self.g.insert(start_idx + i, vec![0.0; param.data.len()]);
            }
            if self.momentum > 0.0 {
                self.buf.insert(start_idx + i, vec![0.0; param.data.len()]);
            }
        }

        // Update params list
        self.params.extend(group.params.clone());

        // Add the group
        self.param_groups.push(group);
    }

    fn state(&self) -> &OptimizerState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut OptimizerState {
        &mut self.state
    }
}

// Implement Send and Sync for thread safety
unsafe impl Send for RMSprop {}
unsafe impl Sync for RMSprop {}
