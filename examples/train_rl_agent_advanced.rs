/// Advanced RL Agent Training with Callbacks
///
/// This example demonstrates production-ready training with:
/// - CSV logging for analysis
/// - Model checkpointing
/// - Early stopping
/// - Learning rate scheduling
/// - Performance monitoring
/// - Console progress updates
use anyhow::Result;
use argus_rl::{
    Action, CSVLogger, CallbackManager, ConsoleLogger, EarlyStopping, EpisodeMetrics,
    LearningRateScheduler, ModelCheckpoint, PerformanceMonitor, ReplayBuffer, RewardCalculator,
    ScheduleType, SdsacTrainer, State, TrainerConfig, TrainingEnvironment, Transition,
};
use rand::Rng;
use std::time::Instant;
use tracing::{info, Level};

const NUM_EPISODES: usize = 1000;
const MAX_STEPS: usize = 50;
const BUFFER_CAPACITY: usize = 10_000;
const WARMUP_STEPS: usize = 500;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("=== Argus RL Agent - Advanced Training ===");
    info!("Episodes: {}", NUM_EPISODES);
    info!("Max steps per episode: {}", MAX_STEPS);
    info!("Replay buffer size: {}", BUFFER_CAPACITY);

    // Create training environment
    let mut env = TrainingEnvironment::new(
        0.6,   // Detection threshold
        false, // Not strict mode
        MAX_STEPS,
    );

    // Create replay buffer
    let mut buffer = ReplayBuffer::new(
        BUFFER_CAPACITY,
        0.6, // Alpha (prioritization)
        0.4, // Beta (importance sampling)
    );

    // Create trainer
    let config = TrainerConfig {
        learning_rate: 3e-4,
        gamma: 0.99,
        tau: 0.005,
        batch_size: 128,
        target_entropy: -8.0,
        grad_clip: 1.0,
    };

    info!("Creating trainer...");
    let mut trainer = SdsacTrainer::new(config)?;

    // Setup callbacks
    let mut callbacks = CallbackManager::new()
        .add(ConsoleLogger::new(10, false)) // Print every 10 episodes
        .add(CSVLogger::new("logs/training_metrics.csv")?)
        .add(ModelCheckpoint::new("checkpoints", 50, true)?) // Save best only
        .add(EarlyStopping::new(100, 1.0)) // Stop if no improvement for 100 episodes
        .add(LearningRateScheduler::new(3e-4, 1e-4, ScheduleType::Cosine))
        .add(PerformanceMonitor::new(20));

    callbacks.on_train_begin(NUM_EPISODES)?;

    // Training loop
    info!("Starting training...");
    let training_start = Instant::now();

    for episode in 0..NUM_EPISODES {
        callbacks.on_episode_begin(episode)?;

        let episode_start = Instant::now();
        let mut episode_metrics = EpisodeMetrics::new(episode);

        // Reset environment
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        // Episode loop
        for step in 0..MAX_STEPS {
            // Select action
            let action = if buffer.len() < WARMUP_STEPS {
                // Random exploration during warmup
                let mut rng = rand::thread_rng();
                Action::all()[rng.gen_range(0..Action::all().len())]
            } else {
                trainer.select_action(&state)?
            };

            // Take step in environment
            let (next_state, reward, done, outcome) = env.step(action)?;

            // Store transition
            buffer.push(Transition {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });

            // Train if enough samples
            if buffer.len() >= WARMUP_STEPS {
                let train_metrics = trainer.train_step(&mut buffer)?;
                callbacks.on_train_step(episode * MAX_STEPS + step, &train_metrics)?;

                // Update episode metrics
                episode_metrics.policy_loss = train_metrics.policy_loss;
                episode_metrics.critic_loss = train_metrics.critic_loss;
                episode_metrics.alpha = train_metrics.alpha;
            }

            total_reward += reward;
            steps += 1;

            if outcome.captcha_detected || outcome.rate_limited {
                episode_metrics.detections += 1;
            }

            state = next_state;

            if done {
                break;
            }
        }

        // Update episode metrics
        episode_metrics.total_reward = total_reward;
        episode_metrics.steps = steps;
        episode_metrics.success_rate = if steps > 0 {
            (steps - episode_metrics.detections) as f32 / steps as f32
        } else {
            0.0
        };
        episode_metrics.duration = episode_start.elapsed();

        // Call episode end callbacks
        let should_continue = callbacks.on_episode_end(episode, &episode_metrics)?;

        if !should_continue {
            info!("Training stopped by callback at episode {}", episode);
            break;
        }
    }

    callbacks.on_train_end()?;

    let total_time = training_start.elapsed();
    info!(
        "Training complete! Total time: {:.1}s ({:.1}m)",
        total_time.as_secs_f32(),
        total_time.as_secs_f32() / 60.0
    );

    // Save final model
    info!("Saving final model...");
    trainer.save("models/sdsac_bot_evasion")?;
    info!("Model saved to models/sdsac_bot_evasion");

    // Print final statistics
    info!("\n=== Final Statistics ===");
    info!("Total episodes trained: {}", NUM_EPISODES);
    info!("Replay buffer size: {}", buffer.len());
    info!("Training logs saved to: logs/training_metrics.csv");
    info!("Model checkpoints saved to: checkpoints/");

    Ok(())
}
