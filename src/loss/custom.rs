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
use crate::loss::bce::Reduction;

pub trait CustomLossFunction {
    fn compute(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError>;
}

pub struct CustomLoss {
    loss_fn: Box<dyn CustomLossFunction>,
    reduction: Reduction,
}

impl CustomLoss {
    pub fn new(loss_fn: Box<dyn CustomLossFunction>, reduction: Reduction) -> Self {
        CustomLoss { loss_fn, reduction }
    }

    pub fn forward(&self, prediction: &Tensor, target: &Tensor) -> Result<Tensor, BellandeError> {
        let loss = self.loss_fn.compute(prediction, target)?;

        match self.reduction {
            Reduction::None => Ok(loss),
            Reduction::Mean => {
                let mean = loss.data.iter().sum::<f32>() / loss.data.len() as f32;
                Ok(Tensor::new(
                    vec![mean],
                    vec![1],
                    true,
                    loss.device,
                    loss.dtype,
                ))
            }
            Reduction::Sum => {
                let sum = loss.data.iter().sum::<f32>();
                Ok(Tensor::new(
                    vec![sum],
                    vec![1],
                    true,
                    loss.device,
                    loss.dtype,
                ))
            }
        }
    }
}
