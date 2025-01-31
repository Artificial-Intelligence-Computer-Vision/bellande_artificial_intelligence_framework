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

pub trait LRScheduler {
    fn step(&mut self);
    fn get_last_lr(&self) -> f32;
}

pub struct StepLR {
    optimizer: Box<dyn Optimizer>,
    step_size: usize,
    gamma: f32,
    base_lr: f32,
    current_step: usize,
}

impl StepLR {
    pub fn new(optimizer: Box<dyn Optimizer>, step_size: usize, gamma: f32) -> Self {
        let base_lr = optimizer.get_lr();
        StepLR {
            optimizer,
            step_size,
            gamma,
            base_lr,
            current_step: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self) {
        self.current_step += 1;
        if self.current_step % self.step_size == 0 {
            let new_lr =
                self.base_lr * self.gamma.powi((self.current_step / self.step_size) as i32);
            self.optimizer.set_lr(new_lr);
        }
    }

    fn get_last_lr(&self) -> f32 {
        self.optimizer.get_lr()
    }
}

pub struct CosineAnnealingLR {
    optimizer: Box<dyn Optimizer>,
    T_max: usize,
    eta_min: f32,
    base_lr: f32,
    current_step: usize,
}

impl CosineAnnealingLR {
    pub fn new(optimizer: Box<dyn Optimizer>, T_max: usize, eta_min: f32) -> Self {
        let base_lr = optimizer.get_lr();
        CosineAnnealingLR {
            optimizer,
            T_max,
            eta_min,
            base_lr,
            current_step: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self) {
        self.current_step += 1;
        let current_step = self.current_step.min(self.T_max);
        let new_lr = self.eta_min
            + (self.base_lr - self.eta_min)
                * (1.0 + std::f32::consts::PI * current_step as f32 / self.T_max as f32).cos()
                / 2.0;
        self.optimizer.set_lr(new_lr);
    }

    fn get_last_lr(&self) -> f32 {
        self.optimizer.get_lr()
    }
}

pub trait Optimizer {
    fn step(&mut self) -> Result<(), BellandeError>;
    fn zero_grad(&mut self);
    fn get_lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}
