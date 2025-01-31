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

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    m: HashMap<usize, Vec<f32>>,
    v: HashMap<usize, Vec<f32>>,
    param_groups: Vec<ParameterGroup>,
    state: OptimizerState,
}

impl Adam {
    pub fn new(
        params: Vec<Tensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let mut m = HashMap::new();
        let mut v = HashMap::new();
        for (idx, param) in params.iter().enumerate() {
            m.insert(idx, vec![0.0; param.data.len()]);
            v.insert(idx, vec![0.0; param.data.len()]);
        }

        // Create default parameter group
        let default_group = ParameterGroup::new(params.clone())
            .with_lr(lr)
            .with_weight_decay(weight_decay)
            .with_betas(betas.0, betas.1)
            .with_eps(eps);

        Adam {
            params,
            lr,
            betas,
            eps,
            weight_decay,
            m,
            v,
            param_groups: vec![default_group],
            state: OptimizerState::new(),
        }
    }
}

// Implement the Optimizer trait for Adam
impl Optimizer for Adam {
    fn step(&mut self) -> Result<(), BellandeError> {
        self.state.increment_step();
        let bias_correction1 = 1.0 - self.betas.0.powi(self.state.step as i32);
        let bias_correction2 = 1.0 - self.betas.1.powi(self.state.step as i32);

        for (idx, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = &param.grad {
                let m = self.m.get_mut(&idx).unwrap();
                let v = self.v.get_mut(&idx).unwrap();

                // Apply updates with proper vectorization
                for ((p, g), (m_i, v_i)) in param
                    .data
                    .iter_mut()
                    .zip(grad.iter())
                    .zip(m.iter_mut().zip(v.iter_mut()))
                {
                    let mut d_p = *g;
                    if self.weight_decay != 0.0 {
                        d_p += self.weight_decay * *p;
                    }

                    // Update biased first moment estimate
                    *m_i = self.betas.0 * *m_i + (1.0 - self.betas.0) * d_p;
                    // Update biased second raw moment estimate
                    *v_i = self.betas.1 * *v_i + (1.0 - self.betas.1) * d_p * d_p;

                    // Compute bias-corrected estimates
                    let m_hat = *m_i / bias_correction1;
                    let v_hat = *v_i / bias_correction2;

                    // Update parameters
                    *p -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
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
        "Adam"
    }

    fn get_param_groups(&self) -> &[ParameterGroup] {
        &self.param_groups
    }

    fn get_param_groups_mut(&mut self) -> &mut [ParameterGroup] {
        &mut self.param_groups
    }

    fn add_param_group(&mut self, mut group: ParameterGroup) {
        let start_idx = self.params.len();

        // Initialize momentum and velocity for new parameters
        for (i, param) in group.params.iter().enumerate() {
            self.m.insert(start_idx + i, vec![0.0; param.data.len()]);
            self.v.insert(start_idx + i, vec![0.0; param.data.len()]);
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
unsafe impl Send for Adam {}
unsafe impl Sync for Adam {}
