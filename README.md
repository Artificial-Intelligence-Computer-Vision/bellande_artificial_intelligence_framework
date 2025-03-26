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
- `cargo add bellande_artificial_intelligence_framework`

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
- [Bellande Git LICENSE](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/LICENSE)
- [GitHub LICENSE](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/LICENSE)
- [GitLab LICENSE](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/bellande_artificial_intelligence_framework/blob/main/LICENSE)
- [BitBucket LICENSE](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/blob/main/LICENSE)

For organizational licensing information, see:
- [Bellande Git Organization Licensing](https://git.bellande-technologies.com/BAICVRI/LICENSING)
- [GitHub Organization Licensing](https://github.com/Artificial-Intelligence-Computer-Vision/LICENSING)
- [GitLab Organization Licensing](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/LICENSING)
- [BitBucket Organization Licensing](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/LICENSING)

## Copyright
Copyright (c) 2024 Bellande Artificial Intelligence Computer Vision Research Institute (BAICVRI)

For copyright details, see:
- [Bellande Git NOTICE](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/NOTICE)
- [GitHub NOTICE](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/NOTICE)
- [GitLab NOTICE](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/bellande_artificial_intelligence_framework/blob/main/NOTICE)
- [BitBucket NOTICE](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/blob/main/NOTICE)

For organizational copyright information, see:
- [Bellande Git Organization Copyright](https://git.bellande-technologies.com/BAICVRI/COPYRIGHT)
- [GitHub Organization Copyright](https://github.com/Artificial-Intelligence-Computer-Vision/COPYRIGHT)
- [GitLab Organization Copyright](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/COPYRIGHT)
- [BitBucket Organization Copyright](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/COPYRIGHT)

## Code of Conduct
We are committed to fostering an open and welcoming environment. For details, see:
- [Bellande Git CODE_OF_CONDUCT](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/CODE_OF_CONDUCT.md)
- [GitHub CODE_OF_CONDUCT](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/CODE_OF_CONDUCT.md)
- [GitLab CODE_OF_CONDUCT](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/bellande_artificial_intelligence_framework/blob/main/CODE_OF_CONDUCT.md)
- [BitBucket CODE_OF_CONDUCT](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/blob/main/CODE_OF_CONDUCT.md)

For organizational code of conduct, see:
- [Bellande Git Organization Code of Conduct](https://git.bellande-technologies.com/BAICVRI/CODE_OF_CONDUCT)
- [GitHub Organization Code of Conduct](https://github.com/Artificial-Intelligence-Computer-Vision/CODE_OF_CONDUCT)
- [GitLab Organization Code of Conduct](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/CODE_OF_CONDUCT)
- [BitBucket Organization Code of Conduct](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/CODE_OF_CONDUCT)

## Terms of Service
By using this framework, you agree to comply with our terms of service. For complete terms, see:
- [Bellande Git TERMS_OF_SERVICE](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/TERMS_OF_SERVICE.md)
- [GitHub TERMS_OF_SERVICE](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/TERMS_OF_SERVICE.md)
- [GitLab TERMS_OF_SERVICE](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/bellande_artificial_intelligence_framework/blob/main/TERMS_OF_SERVICE.md)
- [BitBucket TERMS_OF_SERVICE](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/blob/main/TERMS_OF_SERVICE.md)

For organizational terms of service, see:
- [Bellande Git Organization Profile](https://git.bellande-technologies.com/BAICVRI/.profile)
- [GitHub Organization Profile](https://github.com/Artificial-Intelligence-Computer-Vision/.github)
- [GitLab Organization Profile](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/.gitlab-profile)
- [BitBucket Organization Profile](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/.github)

## Certification
This software has been certified according to our quality standards. For certification details, see:
- [Bellande Git CERTIFICATION](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/CERTIFICATION.md)
- [GitHub CERTIFICATION](https://github.com/BAICVRI/bellande_artificial_intelligence_framework/blob/main/CERTIFICATION.md)
- [GitLab CERTIFICATION](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/bellande_artificial_intelligence_framework/blob/main/CERTIFICATION.md)
- [BitBucket CERTIFICATION](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/blob/main/CERTIFICATION.md)

For organizational certification standards, see:
- [Bellande Git Organization Certification](https://git.bellande-technologies.com/BAICVRI/CERTIFICATION)
- [GitHub Organization Certification](https://github.com/Artificial-Intelligence-Computer-Vision/CERTIFICATION)
- [GitLab Certification](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/CERTIFICATION)
- [BitBucket Certification](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/CERTIFICATION)

## Trademark
For trademark information, see:
- [Bellande Git Organization Trademark](https://git.bellande-technologies.com/BAICVRI/TRADEMARK)
- [GitHub Organization Trademark](https://github.com/Artificial-Intelligence-Computer-Vision/TRADEMARK)
- [GitLab Trademark](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation/TRADEMARK)
- [BitBucket Trademark](https://bitbucket.org/bellande-artificial-intelligence-computer-vision/TRADEMARK)

---

For more information, visit:
- [Bellande Git Organization](https://git.bellande-technologies.com/BAICVRI/bellande_artificial_intelligence_framework)
- [GitHub Organization](https://github.com/BAICVRI/bellande_artificial_intelligence_framework)
- [GitLab Organization](https://gitlab.com/Bellande-Artificial-Intelligence-Computer-Vision-Research-Innovation)
- [BitBucket Organization](https://bitbucket.org/bellande-artificial-intelligence-computer-vision)
