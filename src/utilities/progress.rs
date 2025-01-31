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

use std::io::{stdout, Write};
use std::time::{Duration, Instant};

pub struct ProgressBar {
    total: usize,
    current: usize,
    start_time: Instant,
    last_update: Instant,
    update_frequency: Duration,
}

impl ProgressBar {
    pub fn new(total: usize) -> Self {
        ProgressBar {
            total,
            current: 0,
            start_time: Instant::now(),
            last_update: Instant::now(),
            update_frequency: Duration::from_millis(100),
        }
    }

    pub fn update(&mut self, amount: usize) {
        self.current += amount;
        let now = Instant::now();
        if now.duration_since(self.last_update) >= self.update_frequency {
            self.render();
            self.last_update = now;
        }
    }

    pub fn finish(&mut self) {
        self.current = self.total;
        self.render();
        println!();
    }

    fn render(&self) {
        let progress = self.current as f32 / self.total as f32;
        let bar_width = 50;
        let filled = (progress * bar_width as f32) as usize;
        let empty = bar_width - filled;

        let elapsed = self.start_time.elapsed();
        let eta = if progress > 0.0 {
            Duration::from_secs_f32(elapsed.as_secs_f32() / progress * (1.0 - progress))
        } else {
            Duration::from_secs(0)
        };

        print!(
            "\r[{}{}] {}/{} ({:.1}%) - Elapsed: {:?}, ETA: {:?}",
            "=".repeat(filled),
            " ".repeat(empty),
            self.current,
            self.total,
            progress * 100.0,
            elapsed,
            eta
        );
        stdout().flush().unwrap();
    }
}
