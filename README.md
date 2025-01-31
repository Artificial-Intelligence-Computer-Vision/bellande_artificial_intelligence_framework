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
use bellande_ai_training_framework::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let mut framework = Framework::new()?;
    framework.initialize()?;

    // Create model
    let model = Sequential::new()
        .add(Conv2d::new(3, 64, 3, 1, 1))
        .add(ReLU::new())
        .add(Linear::new(64, 10));

    // Configure training
    let optimizer = Adam::new(model.parameters(), 0.001);
    let loss_fn = CrossEntropyLoss::new();
    let trainer = Trainer::new(model, optimizer, loss_fn);

    // Train model
    trainer.fit(train_loader, Some(val_loader), 100)?;

    Ok(())
}
```

## License
Bellande Artificial Intelligence Training Framework is distributed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html), see [LICENSE](https://github.com/Artificial-Intelligence-Computer-Vision/bellande_artificial_intelligence_training_framework/blob/main/LICENSE) and [NOTICE](https://github.com/Artificial-Intelligence-Computer-Vision/bellande_artificial_intelligence_training_framework/blob/main/LICENSE) for more information.
