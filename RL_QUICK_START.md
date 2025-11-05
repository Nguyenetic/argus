# RL Agent Quick Start Guide 🚀

**Status**: Ready for training and deployment
**Time to first run**: ~10 minutes

---

## 📦 Installation

### 1. Install PyTorch (Required for training)

**Option A: Download libtorch**
```bash
# Visit: https://pytorch.org/get-started/locally/
# Download C++ distribution (libtorch)
# Extract and set environment variable:

# Windows
set LIBTORCH=C:\path\to\libtorch

# Linux/Mac
export LIBTORCH=/path/to/libtorch
```

**Option B: Use Python PyTorch**
```bash
pip install torch torchvision torchaudio

# Set environment variable
export LIBTORCH_USE_PYTORCH=1
```

### 2. Build Argus
```bash
cd web-scraper
cargo build --release
```

---

## 🎓 Training (First Time)

### Quick Training Run
```bash
cargo run --release --example train_rl_agent
```

This will:
- Create synthetic training environment
- Train SDSAC agent for 100 episodes
- Save model to `models/sdsac_bot_evasion`
- Log training metrics

### Training Output
```
Episode 1/100: reward=12.5, steps=45, detections=2
Episode 2/100: reward=18.3, steps=50, detections=1
Episode 3/100: reward=24.7, steps=50, detections=0
...
Episode 100/100: reward=85.2, steps=50, detections=0

Training complete! Model saved to models/sdsac_bot_evasion
```

### Training Time
- **CPU**: ~2-4 hours (100 episodes)
- **GPU**: ~30-60 minutes (100 episodes)
- **Recommended**: 1,000+ episodes for best performance

---

## 🌐 Deployment (Run on Real Sites)

### Quick Demo
```bash
cargo run --release --example rl_agent_demo
```

This will:
- Launch Chromium browser
- Load trained RL agent
- Test on 4 websites:
  - example.com (simple)
  - amazon.com (e-commerce)
  - reddit.com (social media)
  - cloudflare.com (protected)
- Show detection rates and behavior scores

### Demo Output
```
=== Testing: E-commerce (https://www.amazon.com) ===
Running agent...

Session: 20 steps, 0 detections, 95.0% success, 0.87 behavior score, 45.3s duration

Step-by-step breakdown:
  Step  1: WaitShort - wait_short (1240ms) ✓
  Step  2: MouseMovement - mouse_movement (892ms) ✓
  Step  3: ScrollSmall - scroll_small (2341ms) ✓
  Step  4: Interact - interact (3156ms) ✓
  ...

Session Statistics:
  Total actions: 20
  Requests: 4
  Success rate: 95.0%
  Avg interval: 2.26s
  Behavior score: 0.87

Success: YES ✓
```

---

## 🔧 Integration with Your Code

### Basic Usage
```rust
use argus_rl::{IntegratedAgent, AgentConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Launch browser
    let (browser, _) = Browser::launch(config).await?;
    let page = browser.new_page("https://target-site.com").await?;

    // Create agent
    let mut agent = IntegratedAgent::new(
        "models/sdsac_bot_evasion",
        1920.0,  // viewport width
        1080.0,  // viewport height
        AgentConfig::default(),
    )?;

    // Run for 20 steps
    let result = agent.run(&page, 20).await?;

    // Check if successful
    if result.is_successful() {
        println!("Successfully evaded detection!");
    } else {
        println!("Detected {} times", result.detections);
    }

    Ok(())
}
```

### Single Step Usage
```rust
// For fine-grained control
loop {
    let step = agent.step(&page).await?;

    if step.detection.detected {
        println!("Detection: {:?}", step.detection.reason);
        break;
    }

    println!("Action: {:?}, Success: {}",
             step.action, step.result.success);
}
```

---

## 🎛️ Configuration

### Agent Config
```rust
use std::time::Duration;

let config = AgentConfig {
    state_window_size: 50,              // Track last 50 actions
    stop_on_detection: false,            // Continue after detection
    max_duration: Duration::from_secs(300), // 5 minute max
    debug: false,                        // Enable debug logging
};
```

### Trainer Config
```rust
let config = TrainerConfig {
    learning_rate: 3e-4,
    gamma: 0.99,
    tau: 0.005,
    batch_size: 128,
    target_entropy: -8.0,  // Automatic for 8 actions
    grad_clip: 1.0,
};
```

---

## 📊 Understanding Metrics

### Behavior Score (0-1)
- **> 0.8**: Highly human-like ✅
- **0.6-0.8**: Acceptable 👍
- **< 0.6**: Bot-like patterns ⚠️

### Success Rate
- **> 90%**: Excellent ✅
- **70-90%**: Good 👍
- **< 70%**: Needs improvement ⚠️

### Detections
- **0**: Perfect evasion ✅
- **1-2**: Minor issues 👍
- **3+**: High detection rate ⚠️

---

## 🐛 Troubleshooting

### "Cannot find libtorch"
```bash
# Install PyTorch and set environment variable
export LIBTORCH_USE_PYTORCH=1
# OR download libtorch and set path
export LIBTORCH=/path/to/libtorch
```

### "Browser launch failed"
```bash
# Install Chrome/Chromium
# Windows: Download from google.com/chrome
# Linux: sudo apt install chromium-browser
# Mac: brew install chromium
```

### "Model not found"
```bash
# Train the model first
cargo run --example train_rl_agent

# Model will be saved to models/sdsac_bot_evasion
```

### High Detection Rate
- Train for more episodes (1,000+)
- Adjust reward penalties
- Test on different websites
- Check behavior scores

---

## 📁 File Structure

```
web-scraper/
├── crates/argus-rl/         # RL agent crate
│   └── src/
│       ├── state.rs         # 15D state space
│       ├── action.rs        # 8 discrete actions
│       ├── reward.rs        # Reward function
│       ├── networks.rs      # Neural networks
│       ├── buffer.rs        # Replay buffer
│       ├── trainer.rs       # SDSAC algorithm
│       ├── environment.rs   # Training env
│       ├── behavior.rs      # Human emulation
│       ├── executor.rs      # Browser actions
│       ├── integration.rs   # Full integration
│       └── agent.rs         # Agent wrapper
│
├── examples/
│   ├── train_rl_agent.rs   # Training script
│   └── rl_agent_demo.rs    # Deployment demo
│
├── models/                  # Trained models (after training)
│   └── sdsac_bot_evasion/
│
└── RL_AGENT_COMPLETE.md    # Full documentation
```

---

## 🎯 Quick Commands

```bash
# Build
cargo build --release

# Train agent
cargo run --release --example train_rl_agent

# Run demo
cargo run --release --example rl_agent_demo

# Run tests (requires libtorch)
cargo test -p argus-rl

# Check compilation
cargo check -p argus-rl

# Generate docs
cargo doc -p argus-rl --open
```

---

## 📖 Learning Resources

### Understanding the Code
1. **Start with**: `RL_AGENT_COMPLETE.md` (comprehensive overview)
2. **Then read**: `crates/argus-rl/src/lib.rs` (module exports)
3. **Explore**: Individual modules in order:
   - `state.rs` → `action.rs` → `reward.rs`
   - `networks.rs` → `buffer.rs` → `trainer.rs`
   - `behavior.rs` → `executor.rs` → `integration.rs`

### Research Papers
- **SDSAC**: https://arxiv.org/abs/1910.07207
- **SAC**: https://arxiv.org/abs/1801.01290
- **PER**: https://arxiv.org/abs/1511.05952

### Rust Resources
- **tch-rs**: https://github.com/LaurentMazare/tch-rs
- **chromiumoxide**: https://github.com/mattsse/chromiumoxide

---

## 🚀 Next Steps

### 1. Train Your Model
```bash
cargo run --release --example train_rl_agent
```

### 2. Test on Your Site
```rust
let page = browser.new_page("https://your-site.com").await?;
let result = agent.run(&page, 20).await?;
```

### 3. Integrate with Argus
```rust
// In your scraping code
use argus_rl::IntegratedAgent;

let agent = IntegratedAgent::new(...)?;
let result = agent.run(&page, 10).await?;

if result.is_successful() {
    // Proceed with scraping
}
```

### 4. Monitor and Improve
- Track detection rates
- Adjust training episodes
- Tune hyperparameters
- Collect real-world data

---

## 💡 Tips

1. **Start Small**: Train with 100 episodes first
2. **Test Locally**: Use example.com for initial testing
3. **Monitor Metrics**: Watch behavior scores and detection rates
4. **Iterate**: Retrain if detection rate is high
5. **Be Patient**: Good training takes time (1,000+ episodes)

---

## ✅ Success Checklist

- [ ] PyTorch installed
- [ ] Argus compiled successfully
- [ ] Training completed
- [ ] Model saved
- [ ] Demo runs successfully
- [ ] Behavior score > 0.8
- [ ] Detection rate < 5%
- [ ] Ready for production use

---

**Ready to start? Run:** `cargo run --release --example train_rl_agent`

**Questions?** Check `RL_AGENT_COMPLETE.md` for comprehensive documentation.

**Good luck!** 🎉
