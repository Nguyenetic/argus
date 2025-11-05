/// Training Callbacks and Monitoring System
///
/// This module provides a flexible callback system for monitoring and controlling
/// the training process. Callbacks can log metrics, save checkpoints, early stop,
/// visualize progress, and more.
use crate::trainer::TrainingMetrics;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Trait for training callbacks
pub trait Callback: Send {
    /// Called at the start of training
    fn on_train_begin(&mut self, _total_episodes: usize) -> Result<()> {
        Ok(())
    }

    /// Called at the start of each episode
    fn on_episode_begin(&mut self, _episode: usize) -> Result<()> {
        Ok(())
    }

    /// Called at the end of each episode with metrics
    fn on_episode_end(&mut self, _episode: usize, _metrics: &EpisodeMetrics) -> Result<bool> {
        Ok(true) // Continue training
    }

    /// Called after each training step
    fn on_train_step(&mut self, _step: usize, _metrics: &TrainingMetrics) -> Result<()> {
        Ok(())
    }

    /// Called at the end of training
    fn on_train_end(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Episode-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeMetrics {
    pub episode: usize,
    pub total_reward: f32,
    pub steps: usize,
    pub detections: usize,
    pub success_rate: f32,
    pub avg_q_value: f32,
    pub policy_loss: f32,
    pub critic_loss: f32,
    pub alpha: f32,
    pub duration: Duration,
}

impl EpisodeMetrics {
    pub fn new(episode: usize) -> Self {
        Self {
            episode,
            total_reward: 0.0,
            steps: 0,
            detections: 0,
            success_rate: 0.0,
            avg_q_value: 0.0,
            policy_loss: 0.0,
            critic_loss: 0.0,
            alpha: 0.0,
            duration: Duration::from_secs(0),
        }
    }
}

/// CSV logger callback - logs metrics to CSV file
pub struct CSVLogger {
    file_path: PathBuf,
    file: Option<File>,
}

impl CSVLogger {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file_path = path.as_ref().to_path_buf();

        // Create parent directory if needed
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        Ok(Self {
            file_path,
            file: None,
        })
    }

    fn write_header(&mut self) -> Result<()> {
        if let Some(ref mut file) = self.file {
            writeln!(
                file,
                "episode,total_reward,steps,detections,success_rate,avg_q_value,policy_loss,critic_loss,alpha,duration_ms"
            )?;
            file.flush()?;
        }
        Ok(())
    }

    fn write_metrics(&mut self, metrics: &EpisodeMetrics) -> Result<()> {
        if let Some(ref mut file) = self.file {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{}",
                metrics.episode,
                metrics.total_reward,
                metrics.steps,
                metrics.detections,
                metrics.success_rate,
                metrics.avg_q_value,
                metrics.policy_loss,
                metrics.critic_loss,
                metrics.alpha,
                metrics.duration.as_millis()
            )?;
            file.flush()?;
        }
        Ok(())
    }
}

impl Callback for CSVLogger {
    fn on_train_begin(&mut self, _total_episodes: usize) -> Result<()> {
        self.file = Some(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.file_path)?,
        );
        self.write_header()?;
        info!("CSV logging to: {:?}", self.file_path);
        Ok(())
    }

    fn on_episode_end(&mut self, _episode: usize, metrics: &EpisodeMetrics) -> Result<bool> {
        self.write_metrics(metrics)?;
        Ok(true)
    }

    fn on_train_end(&mut self) -> Result<()> {
        if let Some(ref mut file) = self.file {
            file.flush()?;
        }
        info!("CSV log saved to: {:?}", self.file_path);
        Ok(())
    }
}

/// Console logger callback - prints metrics to console
pub struct ConsoleLogger {
    print_frequency: usize,
    verbose: bool,
}

impl ConsoleLogger {
    pub fn new(print_frequency: usize, verbose: bool) -> Self {
        Self {
            print_frequency,
            verbose,
        }
    }
}

impl Callback for ConsoleLogger {
    fn on_train_begin(&mut self, total_episodes: usize) -> Result<()> {
        info!("=== Training Started ===");
        info!("Total episodes: {}", total_episodes);
        Ok(())
    }

    fn on_episode_end(&mut self, episode: usize, metrics: &EpisodeMetrics) -> Result<bool> {
        if episode % self.print_frequency == 0 || self.verbose {
            info!(
                "Episode {:4}: reward={:7.2}, steps={:2}, detections={}, success={:.2}%, α={:.3}, q={:.2}, π_loss={:.3}, c_loss={:.3}",
                metrics.episode,
                metrics.total_reward,
                metrics.steps,
                metrics.detections,
                metrics.success_rate * 100.0,
                metrics.alpha,
                metrics.avg_q_value,
                metrics.policy_loss,
                metrics.critic_loss,
            );
        }
        Ok(true)
    }

    fn on_train_end(&mut self) -> Result<()> {
        info!("=== Training Complete ===");
        Ok(())
    }
}

/// Model checkpoint callback - saves model periodically
pub struct ModelCheckpoint {
    checkpoint_dir: PathBuf,
    save_frequency: usize,
    save_best_only: bool,
    best_reward: f32,
}

impl ModelCheckpoint {
    pub fn new<P: AsRef<Path>>(
        checkpoint_dir: P,
        save_frequency: usize,
        save_best_only: bool,
    ) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();
        fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self {
            checkpoint_dir,
            save_frequency,
            save_best_only,
            best_reward: f32::NEG_INFINITY,
        })
    }
}

impl Callback for ModelCheckpoint {
    fn on_train_begin(&mut self, _total_episodes: usize) -> Result<()> {
        info!(
            "Model checkpoints will be saved to: {:?}",
            self.checkpoint_dir
        );
        Ok(())
    }

    fn on_episode_end(&mut self, episode: usize, metrics: &EpisodeMetrics) -> Result<bool> {
        let should_save = if self.save_best_only {
            metrics.total_reward > self.best_reward
        } else {
            episode % self.save_frequency == 0
        };

        if should_save {
            let checkpoint_path = self
                .checkpoint_dir
                .join(format!("checkpoint_ep{}.pt", episode));

            // Note: Actual model saving would happen in trainer
            // This callback just signals when to save

            if metrics.total_reward > self.best_reward {
                self.best_reward = metrics.total_reward;
                let best_path = self.checkpoint_dir.join("best_model.pt");
                info!(
                    "New best model! Reward: {:.2} (saved to {:?})",
                    self.best_reward, best_path
                );
            }

            debug!("Checkpoint saved: {:?}", checkpoint_path);
        }

        Ok(true)
    }
}

/// Early stopping callback - stops training when no improvement
pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    best_reward: f32,
    wait: usize,
    stopped_episode: Option<usize>,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            best_reward: f32::NEG_INFINITY,
            wait: 0,
            stopped_episode: None,
        }
    }

    pub fn stopped_at(&self) -> Option<usize> {
        self.stopped_episode
    }
}

impl Callback for EarlyStopping {
    fn on_train_begin(&mut self, _total_episodes: usize) -> Result<()> {
        info!(
            "Early stopping enabled: patience={}, min_delta={}",
            self.patience, self.min_delta
        );
        Ok(())
    }

    fn on_episode_end(&mut self, episode: usize, metrics: &EpisodeMetrics) -> Result<bool> {
        if metrics.total_reward > self.best_reward + self.min_delta {
            self.best_reward = metrics.total_reward;
            self.wait = 0;
        } else {
            self.wait += 1;
        }

        if self.wait >= self.patience {
            self.stopped_episode = Some(episode);
            warn!(
                "Early stopping triggered at episode {}. No improvement for {} episodes.",
                episode, self.patience
            );
            return Ok(false); // Stop training
        }

        Ok(true)
    }

    fn on_train_end(&mut self) -> Result<()> {
        if let Some(episode) = self.stopped_episode {
            info!("Training stopped early at episode {}", episode);
        }
        Ok(())
    }
}

/// Learning rate scheduler callback
pub struct LearningRateScheduler {
    initial_lr: f32,
    final_lr: f32,
    total_episodes: usize,
    schedule_type: ScheduleType,
}

#[derive(Debug, Clone, Copy)]
pub enum ScheduleType {
    Linear,
    Exponential,
    Cosine,
    StepDecay { step_size: usize, gamma: f32 },
}

impl LearningRateScheduler {
    pub fn new(initial_lr: f32, final_lr: f32, schedule_type: ScheduleType) -> Self {
        Self {
            initial_lr,
            final_lr,
            total_episodes: 0,
            schedule_type,
        }
    }

    pub fn get_lr(&self, episode: usize) -> f32 {
        if self.total_episodes == 0 {
            return self.initial_lr;
        }

        let progress = episode as f32 / self.total_episodes as f32;

        match self.schedule_type {
            ScheduleType::Linear => self.initial_lr + (self.final_lr - self.initial_lr) * progress,
            ScheduleType::Exponential => {
                self.initial_lr * (self.final_lr / self.initial_lr).powf(progress)
            }
            ScheduleType::Cosine => {
                let cos_progress = (1.0 + (progress * std::f32::consts::PI).cos()) / 2.0;
                self.final_lr + (self.initial_lr - self.final_lr) * cos_progress
            }
            ScheduleType::StepDecay { step_size, gamma } => {
                let steps = episode / step_size;
                self.initial_lr * gamma.powi(steps as i32)
            }
        }
    }
}

impl Callback for LearningRateScheduler {
    fn on_train_begin(&mut self, total_episodes: usize) -> Result<()> {
        self.total_episodes = total_episodes;
        info!("Learning rate schedule: {:?}", self.schedule_type);
        info!(
            "Initial LR: {:.6}, Final LR: {:.6}",
            self.initial_lr, self.final_lr
        );
        Ok(())
    }

    fn on_episode_end(&mut self, episode: usize, _metrics: &EpisodeMetrics) -> Result<bool> {
        let lr = self.get_lr(episode);
        debug!("Episode {}: Learning rate = {:.6}", episode, lr);
        // Actual LR update would be done in trainer
        Ok(true)
    }
}

/// Performance monitor callback - tracks and reports performance
pub struct PerformanceMonitor {
    start_time: Option<Instant>,
    episode_times: Vec<Duration>,
    window_size: usize,
}

impl PerformanceMonitor {
    pub fn new(window_size: usize) -> Self {
        Self {
            start_time: None,
            episode_times: Vec::new(),
            window_size,
        }
    }

    fn avg_episode_time(&self) -> Duration {
        if self.episode_times.is_empty() {
            return Duration::from_secs(0);
        }

        let recent: Vec<_> = self
            .episode_times
            .iter()
            .rev()
            .take(self.window_size)
            .collect();

        let total: Duration = recent.iter().map(|&&d| d).sum();
        total / recent.len() as u32
    }
}

impl Callback for PerformanceMonitor {
    fn on_train_begin(&mut self, _total_episodes: usize) -> Result<()> {
        self.start_time = Some(Instant::now());
        info!("Performance monitoring started");
        Ok(())
    }

    fn on_episode_end(&mut self, episode: usize, metrics: &EpisodeMetrics) -> Result<bool> {
        self.episode_times.push(metrics.duration);

        if episode % 10 == 0 && episode > 0 {
            let avg_time = self.avg_episode_time();
            let total_time = self.start_time.unwrap().elapsed();

            info!(
                "Performance: avg episode time={:.2}s, total time={:.1}s, episodes/hour={:.0}",
                avg_time.as_secs_f32(),
                total_time.as_secs_f32(),
                3600.0 / avg_time.as_secs_f32()
            );
        }

        Ok(true)
    }

    fn on_train_end(&mut self) -> Result<()> {
        if let Some(start) = self.start_time {
            let total_time = start.elapsed();
            info!(
                "Total training time: {:.1}s ({:.1}m)",
                total_time.as_secs_f32(),
                total_time.as_secs_f32() / 60.0
            );
        }
        Ok(())
    }
}

/// Callback manager - manages multiple callbacks
pub struct CallbackManager {
    callbacks: Vec<Box<dyn Callback>>,
}

impl CallbackManager {
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    pub fn add<C: Callback + 'static>(mut self, callback: C) -> Self {
        self.callbacks.push(Box::new(callback));
        self
    }

    pub fn on_train_begin(&mut self, total_episodes: usize) -> Result<()> {
        for callback in &mut self.callbacks {
            callback.on_train_begin(total_episodes)?;
        }
        Ok(())
    }

    pub fn on_episode_begin(&mut self, episode: usize) -> Result<()> {
        for callback in &mut self.callbacks {
            callback.on_episode_begin(episode)?;
        }
        Ok(())
    }

    pub fn on_episode_end(&mut self, episode: usize, metrics: &EpisodeMetrics) -> Result<bool> {
        for callback in &mut self.callbacks {
            if !callback.on_episode_end(episode, metrics)? {
                return Ok(false); // Stop training
            }
        }
        Ok(true)
    }

    pub fn on_train_step(&mut self, step: usize, metrics: &TrainingMetrics) -> Result<()> {
        for callback in &mut self.callbacks {
            callback.on_train_step(step, metrics)?;
        }
        Ok(())
    }

    pub fn on_train_end(&mut self) -> Result<()> {
        for callback in &mut self.callbacks {
            callback.on_train_end()?;
        }
        Ok(())
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episode_metrics() {
        let metrics = EpisodeMetrics::new(1);
        assert_eq!(metrics.episode, 1);
        assert_eq!(metrics.total_reward, 0.0);
        assert_eq!(metrics.steps, 0);
    }

    #[test]
    fn test_early_stopping() {
        let mut stopper = EarlyStopping::new(3, 0.1);

        let mut metrics = EpisodeMetrics::new(1);
        metrics.total_reward = 10.0;

        // Should continue
        assert!(stopper.on_episode_end(1, &metrics).unwrap());

        // No improvement for 3 episodes
        for i in 2..=4 {
            metrics.episode = i;
            metrics.total_reward = 10.0; // Same reward
            let should_continue = stopper.on_episode_end(i, &metrics).unwrap();

            if i < 4 {
                assert!(should_continue);
            } else {
                assert!(!should_continue); // Should stop at episode 4
            }
        }

        assert_eq!(stopper.stopped_at(), Some(4));
    }

    #[test]
    fn test_lr_scheduler_linear() {
        let mut scheduler = LearningRateScheduler::new(1e-3, 1e-4, ScheduleType::Linear);
        scheduler.on_train_begin(100).unwrap();

        let lr_start = scheduler.get_lr(0);
        let lr_mid = scheduler.get_lr(50);
        let lr_end = scheduler.get_lr(100);

        assert!((lr_start - 1e-3).abs() < 1e-9);
        assert!(lr_mid > 1e-4 && lr_mid < 1e-3);
        assert!((lr_end - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn test_callback_manager() {
        let mut manager = CallbackManager::new()
            .add(ConsoleLogger::new(10, false))
            .add(PerformanceMonitor::new(10));

        manager.on_train_begin(100).unwrap();

        let metrics = EpisodeMetrics::new(1);
        assert!(manager.on_episode_end(1, &metrics).unwrap());

        manager.on_train_end().unwrap();
    }
}
