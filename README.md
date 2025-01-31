# Bellande Artificial Intelligence Training Framework

Bellande training framework in Rust for machine learning models

# Run Bellos Scripts
    - build_bellande_framework.bellos
    - make_rust_executable.bellos
    - run_bellande_framework.bellos

# Run Bash Scripts
    - build_bellande_framework.sh
    - make_rust_executable.sh
    - run_bellande_framework.sh

# Testing
- "cargo test" for a quick test

## Example Usage
```rust
use bellande_artificial_intelligence_training_framework::{
    core::tensor::Tensor,
    layer::{activation::ReLU, conv::Conv2d},
    models::sequential::Sequential,
};
use std::error::Error;

// Simple single-layer model example
fn main() -> Result> {
    // Create a simple sequential model
    let mut model = Sequential::new();
    
    // Add a convolutional layer
    model.add(Box::new(Conv2d::new(
        3,            // input channels
        4,            // output channels
        (3, 3),       // kernel size
        Some((1, 1)), // stride
        Some((1, 1)), // padding
        true,         // use bias
    )));
    
    // Create input tensor
    let input = Tensor::zeros(&[1, 3, 8, 8]); // batch_size=1, channels=3, height=8, width=8
    
    // Forward pass
    let output = model.forward(&input)?;
    
    // Print output shape
    println!("Output shape: {:?}", output.shape());
    assert_eq!(output.shape()[1], 4); // Verify output channels
    
    Ok(())
}
```

## License
Bellande Artificial Intelligence Training Framework is distributed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html), see [LICENSE](https://github.com/Artificial-Intelligence-Computer-Vision/bellande_artificial_intelligence_training_framework/blob/main/LICENSE) and [NOTICE](https://github.com/Artificial-Intelligence-Computer-Vision/bellande_artificial_intelligence_training_framework/blob/main/LICENSE) for more information.
