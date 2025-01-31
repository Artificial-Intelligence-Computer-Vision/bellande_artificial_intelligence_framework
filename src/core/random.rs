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

use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use std::cell::RefCell;

thread_local! {
    static GENERATOR: RefCell<StdRng> = RefCell::new(StdRng::from_entropy());
}

pub fn set_seed(seed: u64) {
    GENERATOR.with(|g| {
        *g.borrow_mut() = StdRng::seed_from_u64(seed);
    });
}

pub fn normal(mean: f32, std: f32, size: usize) -> Vec<f32> {
    let normal = Normal::new(mean as f64, std as f64).unwrap();
    GENERATOR.with(|g| {
        (0..size)
            .map(|_| normal.sample(&mut *g.borrow_mut()) as f32)
            .collect()
    })
}

pub fn uniform(low: f32, high: f32, size: usize) -> Vec<f32> {
    let uniform = Uniform::new(low, high);
    GENERATOR.with(|g| {
        (0..size)
            .map(|_| uniform.sample(&mut *g.borrow_mut()))
            .collect()
    })
}

pub fn bernoulli(p: f32, size: usize) -> Vec<bool> {
    GENERATOR.with(|g| (0..size).map(|_| g.borrow_mut().gen::<f32>() < p).collect())
}
