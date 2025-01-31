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

use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

pub struct VisualizationBuilder {
    title: String,
    width: u32,
    height: u32,
    x_label: String,
    y_label: String,
}

impl VisualizationBuilder {
    pub fn new() -> Self {
        VisualizationBuilder {
            title: String::from("Plot"),
            width: 800,
            height: 600,
            x_label: String::from("X"),
            y_label: String::from("Y"),
        }
    }

    pub fn title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn labels(mut self, x_label: &str, y_label: &str) -> Self {
        self.x_label = x_label.to_string();
        self.y_label = y_label.to_string();
        self
    }
}

pub struct Visualization;

impl Visualization {
    pub fn plot_metrics<P: AsRef<Path>>(
        history: &HashMap<String, Vec<f32>>,
        metrics: &[&str],
        output_path: P,
        config: VisualizationBuilder,
    ) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(output_path.as_ref(), (config.width, config.height))
            .into_drawing_area();

        root.fill(&WHITE)?;

        let epochs: Vec<f32> = (0..history.values().next().unwrap().len())
            .map(|x| x as f32)
            .collect();

        let min_value = metrics
            .iter()
            .filter_map(|&m| history.get(m))
            .flatten()
            .fold(f32::INFINITY, |a, &b| a.min(b));

        let max_value = metrics
            .iter()
            .filter_map(|&m| history.get(m))
            .flatten()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption(&config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0f32..epochs.len() as f32, min_value..max_value)?;

        chart
            .configure_mesh()
            .x_desc(&config.x_label)
            .y_desc(&config.y_label)
            .draw()?;

        for (idx, &metric) in metrics.iter().enumerate() {
            if let Some(values) = history.get(metric) {
                chart
                    .draw_series(LineSeries::new(
                        epochs.iter().zip(values).map(|(&x, &y)| (x, y)),
                        &Palette99::pick(idx),
                    ))?
                    .label(metric)
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], &Palette99::pick(idx))
                    });
            }
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        Ok(())
    }

    pub fn plot_confusion_matrix<P: AsRef<Path>>(
        matrix: &Vec<Vec<usize>>,
        labels: &[String],
        output_path: P,
    ) -> Result<(), Box<dyn Error>> {
        let width = 800;
        let height = 600;
        let root = BitMapBackend::new(output_path.as_ref(), (width, height)).into_drawing_area();

        root.fill(&WHITE)?;

        let n_classes = matrix.len();
        let max_value = matrix.iter().flatten().max().copied().unwrap_or(1);

        let mut chart = ChartBuilder::on(&root)
            .caption("Confusion Matrix", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0f32..n_classes as f32, 0f32..n_classes as f32)?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .x_desc("Predicted")
            .y_desc("Actual")
            .x_labels(n_classes)
            .y_labels(n_classes)
            .x_label_formatter(&|x| labels[x.floor() as usize].clone())
            .y_label_formatter(&|y| labels[y.floor() as usize].clone())
            .draw()?;

        for i in 0..n_classes {
            for j in 0..n_classes {
                let value = matrix[i][j];
                let color = RGBColor(
                    255,
                    ((1.0 - value as f64 / max_value as f64) * 255.0) as u8,
                    ((1.0 - value as f64 / max_value as f64) * 255.0) as u8,
                );

                // Use f32 for coordinates
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(j as f32, i as f32), ((j + 1) as f32, (i + 1) as f32)],
                    color.filled(),
                )))?;

                chart.draw_series(std::iter::once(Text::new(
                    value.to_string(),
                    (j as f32 + 0.5, i as f32 + 0.5),
                    ("sans-serif", 20).into_font(),
                )))?;
            }
        }

        Ok(())
    }
}
