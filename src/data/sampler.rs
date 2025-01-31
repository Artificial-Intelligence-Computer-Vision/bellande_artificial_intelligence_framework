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

use rand::seq::SliceRandom;
use std::sync::atomic::{AtomicUsize, Ordering};

pub trait Sampler: Send + Sync {
    fn sample(&mut self, n: usize) -> Vec<usize>;
    fn len(&self) -> usize;
}

pub struct RandomSampler {
    data_len: usize,
    current_index: AtomicUsize,
    indices: Vec<usize>,
}

impl RandomSampler {
    pub fn new(data_len: usize) -> Self {
        let mut indices: Vec<usize> = (0..data_len).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        RandomSampler {
            data_len,
            current_index: AtomicUsize::new(0),
            indices,
        }
    }
}

impl Sampler for RandomSampler {
    fn sample(&mut self, n: usize) -> Vec<usize> {
        let current = self.current_index.fetch_add(n, Ordering::SeqCst);
        if current >= self.data_len {
            let mut indices: Vec<usize> = (0..self.data_len).collect();
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
            self.indices = indices;
            self.current_index.store(n, Ordering::SeqCst);
            self.indices[0..n].to_vec()
        } else {
            self.indices[current..current + n.min(self.data_len - current)].to_vec()
        }
    }

    fn len(&self) -> usize {
        self.data_len
    }
}

pub struct SequentialSampler {
    data_len: usize,
    current_index: AtomicUsize,
}

impl SequentialSampler {
    pub fn new(data_len: usize) -> Self {
        SequentialSampler {
            data_len,
            current_index: AtomicUsize::new(0),
        }
    }
}

impl Sampler for SequentialSampler {
    fn sample(&mut self, n: usize) -> Vec<usize> {
        let current = self.current_index.fetch_add(n, Ordering::SeqCst);
        if current >= self.data_len {
            self.current_index.store(n, Ordering::SeqCst);
            (0..n.min(self.data_len)).collect()
        } else {
            (current..current + n.min(self.data_len - current)).collect()
        }
    }

    fn len(&self) -> usize {
        self.data_len
    }
}
