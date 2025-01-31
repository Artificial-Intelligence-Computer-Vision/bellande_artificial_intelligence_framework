# Bellande Artificial Intelligence Framework

Bellande artificial intelligence framework in Rust for machine learning models

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
use bellande_artificial_intelligence_framework::{
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

## Website Crates
- https://crates.io/crates/bellande_artificial_intelligence_framework

### Installation
- `cargo add bellande_mesh_sync`

```
Name: Bellande Artificial Intelligence Framework
Summary: Bellande Operating System Comprehensive data synchronization system
Company-Page: git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework
Home-page: github.com/Architecture-Mechanism/bellande_artificial_intelligence_framework
Author: Ronaldson Bellande
Author-email: ronaldsonbellande@gmail.com
License: GNU General Public License v3.0
```

# Legal Documentation

## License
Bellande Artificial Intelligence Framework is distributed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

For detailed license information, see:
- [GitHub LICENSE](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/LICENSE)
- [Bellande LICENSE](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/LICENSE)

For organizational licensing information, see:
- [GitHub Organization Licensing](https://github.com/Artificial-Intelligence-Computer-Vision/LICENSING)
- [Bellande Organization Licensing](https://git.bellande-technologies.com/BAICVRI/LICENSING)

## Copyright
Copyright (c) 2024 Bellande Artificial Intelligence Computer Vision Research Institute (BAICVRI)

For copyright details, see:
- [GitHub NOTICE](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/NOTICE)
- [Bellande NOTICE](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/NOTICE)

For organizational copyright information, see:
- [GitHub Organization Copyright](https://github.com/Artificial-Intelligence-Computer-Vision/COPYRIGHT)
- [Bellande Organization Copyright](https://git.bellande-technologies.com/BAICVRI/COPYRIGHT)

## Code of Conduct
We are committed to fostering an open and welcoming environment. For details, see:
- [GitHub CODE_OF_CONDUCT](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/CODE_OF_CONDUCT.md)
- [Bellande CODE_OF_CONDUCT](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/CODE_OF_CONDUCT.md)

For organizational code of conduct, see:
- [GitHub Organization Code of Conduct](https://github.com/Artificial-Intelligence-Computer-Vision/CODE_OF_CONDUCT)
- [Bellande Organization Code of Conduct](https://git.bellande-technologies.com/BAICVRI/CODE_OF_CONDUCT)

## Terms of Service
By using this framework, you agree to comply with our terms of service. For complete terms, see:
- [GitHub TERMS_OF_SERVICE](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/TERMS_OF_SERVICE.md)
- [Bellande TERMS_OF_SERVICE](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/TERMS_OF_SERVICE.md)

For organizational terms of service, see:
- [GitHub Organization Profile](https://github.com/Artificial-Intelligence-Computer-Vision/.github)
- [Bellande Organization Profile](https://git.bellande-technologies.com/BAICVRI/.profile)

## Certification
This software has been certified according to our quality standards. For certification details, see:
- [GitHub CERTIFICATION](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/CERTIFICATION.md)
- [Bellande CERTIFICATION](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/CERTIFICATION.md)

For organizational certification standards, see:
- [GitHub Organization Certification](https://github.com/Artificial-Intelligence-Computer-Vision/CERTIFICATION)
- [Bellande Organization Certification](https://git.bellande-technologies.com/BAICVRI/CERTIFICATION)

## Trademark
For trademark information, see:
- [GitHub Organization Trademark](https://github.com/Artificial-Intelligence-Computer-Vision/TRADEMARK)
- [Bellande Organization Trademark](https://git.bellande-technologies.com/BAICVRI/TRADEMARK)

---

For more information, visit:
- [GitHub Repository](https://github.com/BAICVRI/bellande_artificial_intelligence_framework)
- [Bellande Repository](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework)
- [GitHub Organization](https://github.com/Artificial-Intelligence-Computer-Vision)
- [Bellande Organization](https://git.bellande-technologies.com/BAICVRI)
