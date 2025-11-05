//! Example: Train the SDSAC RL agent on synthetic bot detection
//!
//! This demonstrates a complete training loop with the RL agent learning
//! to evade a synthetic bot detector.
//!
//! Run with: cargo run --example train_rl_agent --release

use anyhow::Result;
use argus_rl::{Action, ReplayBuffer, SdsacTrainer, TrainerConfig, TrainingEnvironment};
use tch::Device;
use tracing::{info, Level};
use tracing_subscriber;

fn main() -> Result<()> {
    // Setup logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 Starting SDSAC Training Demo");
    info!("================================");

    // Training hyperparameters
    const NUM_EPISODES: usize = 100;
    const MAX_STEPS_PER_EPISODE: usize = 50;
    const BUFFER_CAPACITY: usize = 10_000;
    const BATCH_SIZE: usize = 64;

    // Create environment with moderate difficulty
    let mut env = TrainingEnvironment::new(
        0.6,   // detection_threshold (0.6 = moderate)
        false, // strict_mode = false (lenient)
        MAX_STEPS_PER_EPISODE,
    );

    // Create replay buffer
    let mut buffer = ReplayBuffer::new(
        BUFFER_CAPACITY,
        0.6, // alpha (prioritization)
        0.4, // beta (importance sampling)
    );

    // Create trainer
    let config = TrainerConfig {
        actor_lr: 3e-4,
        critic_lr: 3e-4,
        alpha_lr: 3e-4,
        gamma: 0.99,
        tau: 0.005,
        batch_size: BATCH_SIZE,
        min_buffer_size: 500,
        q_clip: Some((-10.0, 10.0)),
        grad_clip: Some(1.0),
        device: Device::Cpu,
    };

    info!("Configuration:");
    info!("  Episodes: {}", NUM_EPISODES);
    info!("  Max steps/episode: {}", MAX_STEPS_PER_EPISODE);
    info!("  Buffer capacity: {}", BUFFER_CAPACITY);
    info!("  Batch size: {}", BATCH_SIZE);
    info!("  Device: {:?}", config.device);
    info!("");

    let mut trainer = SdsacTrainer::new(config)?;

    // Training metrics
    let mut total_steps = 0;
    let mut successful_episodes = 0;

    info!("🎯 Starting training...");
    info!("");

    // Training loop
    for episode in 0..NUM_EPISODES {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut episode_steps = 0;
        let mut detected = false;

        // Episode loop
        for step in 0..MAX_STEPS_PER_EPISODE {
            // Select action
            let action = trainer.select_action(&state)?;

            // Take step in environment
            let (next_state, reward, done, outcome) = env.step(action)?;

            // Store transition in replay buffer
            buffer.push(argus_rl::Transition {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
                priority: 1.0, // Will be updated based on TD-error
            });

            // Train if buffer has enough samples
            if buffer.len() >= 500 && step % 4 == 0 {
                let metrics = trainer.train_step(&mut buffer)?;

                // Log occasionally
                if total_steps % 100 == 0 && total_steps > 0 {
                    metrics.log();
                }
            }

            episode_reward += reward;
            episode_steps += 1;
            total_steps += 1;
            state = next_state;

            if outcome.access_denied || outcome.captcha_detected {
                detected = true;
            }

            if done {
                break;
            }
        }

        if !detected {
            successful_episodes += 1;
        }

        // Log episode summary
        if episode % 10 == 0 || episode < 5 {
            let success_rate = (successful_episodes as f32 / (episode + 1) as f32) * 100.0;
            info!(
                "Episode {}/{}: Reward={:.2}, Steps={}, Success_Rate={:.1}%, Buffer={}",
                episode + 1,
                NUM_EPISODES,
                episode_reward,
                episode_steps,
                success_rate,
                buffer.len()
            );
        }
    }

    // Final statistics
    info!("");
    info!("✅ Training complete!");
    info!("================================");
    let final_success_rate = (successful_episodes as f32 / NUM_EPISODES as f32) * 100.0;
    info!("Final success rate: {:.1}%", final_success_rate);
    info!("Total steps: {}", total_steps);
    info!(
        "Successful episodes: {}/{}",
        successful_episodes, NUM_EPISODES
    );

    // Save trained model
    trainer.save("models/sdsac_bot_evasion")?;
    info!("💾 Model saved to models/sdsac_bot_evasion_*.pt");

    Ok(())
}
