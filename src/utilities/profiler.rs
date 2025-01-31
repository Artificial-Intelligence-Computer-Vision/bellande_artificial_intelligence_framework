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

use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct Profiler {
    timings: HashMap<String, Vec<Duration>>,
    current_timers: HashMap<String, Instant>,
}

impl Profiler {
    pub fn new() -> Self {
        Profiler {
            timings: HashMap::new(),
            current_timers: HashMap::new(),
        }
    }

    pub fn start(&mut self, name: &str) {
        self.current_timers.insert(name.to_string(), Instant::now());
    }

    pub fn stop(&mut self, name: &str) {
        if let Some(start_time) = self.current_timers.remove(name) {
            let duration = start_time.elapsed();
            self.timings
                .entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    pub fn get_statistics(&self, name: &str) -> Option<ProfileStats> {
        self.timings.get(name).map(|durations| {
            let total: Duration = durations.iter().sum();
            let avg = total / durations.len() as u32;
            let min = durations.iter().min().unwrap();
            let max = durations.iter().max().unwrap();

            ProfileStats {
                count: durations.len(),
                total,
                average: avg,
                min: *min,
                max: *max,
            }
        })
    }

    pub fn reset(&mut self) {
        self.timings.clear();
        self.current_timers.clear();
    }

    pub fn report(&self) -> String {
        let mut report = String::from("Performance Profile:\n");
        for (name, stats) in self.timings.iter() {
            let total: Duration = stats.iter().sum();
            let avg = total / stats.len() as u32;
            report.push_str(&format!(
                "{}: {} calls, total={:?}, avg={:?}\n",
                name,
                stats.len(),
                total,
                avg
            ));
        }
        report
    }
}

#[derive(Debug)]
pub struct ProfileStats {
    pub count: usize,
    pub total: Duration,
    pub average: Duration,
    pub min: Duration,
    pub max: Duration,
}
