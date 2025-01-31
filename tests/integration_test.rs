// Copyright (C) 2025 Bellande Artificial Intelligence Computer Vision Research Innovation Center, Ronaldson Bellande

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

use std::error::Error;

use bellande_artificial_intelligence_training_framework::{
    core::tensor::Tensor,
    layer::{activation::ReLU, conv::Conv2d, linear::Linear, pooling::MaxPool2d},
    models::sequential::Sequential,
};

#[test]
fn test_single_layer() -> Result<(), Box<dyn Error>> {
    // Create the simplest possible model
    let mut model = Sequential::new();

    // Add just one conv layer
    model.add(Box::new(Conv2d::new(
        3,            // in_channels
        4,            // out_channels (reduced)
        (3, 3),       // kernel_size
        Some((1, 1)), // stride
        Some((1, 1)), // padding
        true,         // bias
    )));

    // Create tiny input
    let input = Tensor::zeros(&[1, 3, 8, 8]); // Minimal size

    // Test forward pass
    let output = model.forward(&input)?;

    // Verify output
    assert_eq!(output.shape()[1], 4); // Check output channels

    Ok(())
}

// Test tensor operations separately
#[test]
fn test_tensor_ops() -> Result<(), Box<dyn Error>> {
    let tensor = Tensor::zeros(&[1, 3, 8, 8]);
    assert_eq!(tensor.shape(), &[1, 3, 8, 8]);
    Ok(())
}
