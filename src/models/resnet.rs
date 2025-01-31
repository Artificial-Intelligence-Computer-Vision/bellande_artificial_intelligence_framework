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
use crate::layer::{
    activation::{Activation, ReLU},
    avgpool2d::AvgPool2d,
    batch_norm::BatchNorm2d,
    conv::Conv2d,
    linear::Linear,
    pooling::MaxPool2d,
};
use crate::models::sequential::{NeuralLayer, Sequential};

pub struct ResidualBlock {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    conv2: Conv2d,
    bn2: BatchNorm2d,
    downsample: Option<Sequential>,
    relu: ReLU,
}

impl ResidualBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: Option<Sequential>,
    ) -> Self {
        ResidualBlock {
            conv1: Conv2d::new(
                in_channels,
                out_channels,
                (3, 3),
                Some((stride, stride)),
                Some((1, 1)),
                true,
            ),
            bn1: BatchNorm2d::new(out_channels, 1e-5, 0.1, true),
            conv2: Conv2d::new(
                out_channels,
                out_channels,
                (3, 3),
                Some((1, 1)),
                Some((1, 1)),
                true,
            ),
            bn2: BatchNorm2d::new(out_channels, 1e-5, 0.1, true),
            downsample,
            relu: ReLU::new(),
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor, BellandeError> {
        let identity = if let Some(ref mut ds) = self.downsample {
            ds.forward(x)?
        } else {
            x.clone()
        };

        let mut out = NeuralLayer::forward(&mut self.conv1, x)?;
        out = NeuralLayer::forward(&mut self.bn1, &out)?;
        out = Activation::forward(&self.relu, &out)?;

        out = NeuralLayer::forward(&mut self.conv2, &out)?;
        out = NeuralLayer::forward(&mut self.bn2, &out)?;

        // Use element-wise addition
        out = out.add(&identity)?;
        out = Activation::forward(&self.relu, &out)?;

        Ok(out)
    }
}

pub struct ResNet {
    conv1: Conv2d,
    bn1: BatchNorm2d,
    relu: ReLU,
    maxpool: MaxPool2d,
    layer1: Vec<ResidualBlock>,
    layer2: Vec<ResidualBlock>,
    layer3: Vec<ResidualBlock>,
    layer4: Vec<ResidualBlock>,
    avgpool: AvgPool2d,
    fc: Linear,
}

impl ResNet {
    pub fn resnet18(num_classes: usize) -> Self {
        ResNet {
            conv1: Conv2d::new(3, 64, (7, 7), Some((2, 2)), Some((3, 3)), true),
            bn1: BatchNorm2d::new(64, 1e-5, 0.1, true),
            relu: ReLU::new(),
            maxpool: MaxPool2d::new((3, 3), Some((2, 2))),
            layer1: make_layer(64, 64, 2, 1),
            layer2: make_layer(64, 128, 2, 2),
            layer3: make_layer(128, 256, 2, 2),
            layer4: make_layer(256, 512, 2, 2),
            avgpool: AvgPool2d::new((7, 7), Some((1, 1)), None),
            fc: Linear::new(512, num_classes, true),
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor, BellandeError> {
        let mut out = NeuralLayer::forward(&mut self.conv1, x)?;
        out = NeuralLayer::forward(&mut self.bn1, &out)?;
        out = Activation::forward(&self.relu, &out)?;
        out = NeuralLayer::forward(&mut self.maxpool, &out)?;

        for block in &mut self.layer1 {
            out = block.forward(&out)?;
        }
        for block in &mut self.layer2 {
            out = block.forward(&out)?;
        }
        for block in &mut self.layer3 {
            out = block.forward(&out)?;
        }
        for block in &mut self.layer4 {
            out = block.forward(&out)?;
        }

        out = NeuralLayer::forward(&mut self.avgpool, &out)?;

        // Calculate flattened size for reshape
        let batch_size = out.shape[0];
        let total_features = out.data.len() / batch_size;
        out = out.reshape(&[batch_size, total_features])?;

        out = NeuralLayer::forward(&mut self.fc, &out)?;

        Ok(out)
    }
}

fn make_layer(
    in_channels: usize,
    out_channels: usize,
    blocks: usize,
    stride: usize,
) -> Vec<ResidualBlock> {
    let mut layers = Vec::new();

    let downsample = if stride != 1 || in_channels != out_channels {
        let mut sequential = Sequential::new();
        sequential.add(Box::new(Conv2d::new(
            in_channels,
            out_channels,
            (1, 1),
            Some((stride, stride)),
            Some((0, 0)),
            true,
        )));
        sequential.add(Box::new(BatchNorm2d::new(out_channels, 1e-5, 0.1, true)));
        Some(sequential)
    } else {
        None
    };

    layers.push(ResidualBlock::new(
        in_channels,
        out_channels,
        stride,
        downsample,
    ));

    for _ in 1..blocks {
        layers.push(ResidualBlock::new(out_channels, out_channels, 1, None));
    }

    layers
}

// Implement Send and Sync for thread safety
unsafe impl Send for ResNet {}
unsafe impl Sync for ResNet {}
