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

use crate::core::tensor::Tensor;

pub trait Metric {
    fn reset(&mut self);
    fn update(&mut self, prediction: &Tensor, target: &Tensor);
    fn compute(&self) -> f32;
    fn name(&self) -> &str;
}

pub struct Accuracy {
    correct: usize,
    total: usize,
}

impl Accuracy {
    pub fn new() -> Self {
        Accuracy {
            correct: 0,
            total: 0,
        }
    }
}

impl Metric for Accuracy {
    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn update(&mut self, prediction: &Tensor, target: &Tensor) {
        let pred_classes: Vec<usize> = prediction
            .data
            .chunks(prediction.shape[1])
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        for (pred, &true_class) in pred_classes.iter().zip(target.data.iter()) {
            if *pred == true_class as usize {
                self.correct += 1;
            }
            self.total += 1;
        }
    }

    fn compute(&self) -> f32 {
        self.correct as f32 / self.total as f32
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}
