/// Browser pool manager for parallel scraping
///
/// Manages a pool of browser instances with:
/// - Dynamic scaling (min/max instances)
/// - Health checking and auto-recovery
/// - Load balancing
/// - Resource cleanup
/// - Stealth mode integration
use anyhow::{bail, Context as AnyhowContext, Result};
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::handler::HandlerConfig;
use chromiumoxide::Page;
use futures::stream::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, warn};

use crate::stealth::{StealthConfig, StealthMode, UserAgentManager};

/// Browser pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of browser instances
    pub min_instances: usize,

    /// Maximum number of browser instances
    pub max_instances: usize,

    /// Initial number of browser instances
    pub initial_instances: usize,

    /// Maximum idle time before closing browser (seconds)
    pub max_idle_time: Duration,

    /// Health check interval (seconds)
    pub health_check_interval: Duration,

    /// Enable stealth mode
    pub enable_stealth: bool,

    /// Stealth configuration
    pub stealth_config: StealthConfig,

    /// Browser window size
    pub window_size: (u32, u32),

    /// Headless mode
    pub headless: bool,

    /// Enable incognito mode
    pub incognito: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_instances: 1,
            max_instances: 10,
            initial_instances: 2,
            max_idle_time: Duration::from_secs(300), // 5 minutes
            health_check_interval: Duration::from_secs(30),
            enable_stealth: true,
            stealth_config: StealthConfig::default(),
            window_size: (1920, 1080),
            headless: true,
            incognito: true,
        }
    }
}

/// Browser instance wrapper
struct BrowserInstance {
    id: String,
    browser: Browser,
    created_at: Instant,
    last_used: Instant,
    page_count: usize,
    is_healthy: bool,
    stealth_mode: Option<Arc<StealthMode>>,
}

impl BrowserInstance {
    /// Create new browser instance
    async fn new(
        id: String,
        config: &PoolConfig,
        stealth_mode: Option<Arc<StealthMode>>,
    ) -> Result<Self> {
        let browser_config = BrowserConfig::builder()
            .window_size(config.window_size.0, config.window_size.1)
            .viewport(chromiumoxide::handler::viewport::Viewport {
                width: config.window_size.0,
                height: config.window_size.1,
                ..Default::default()
            })
            .build()
            .context("Failed to build browser config")?;

        let (browser, mut handler) = Browser::launch(browser_config)
            .await
            .context("Failed to launch browser")?;

        // Spawn handler task
        tokio::spawn(async move {
            while let Some(event) = handler.next().await {
                if let Err(e) = event {
                    error!("Browser handler error: {}", e);
                }
            }
        });

        let now = Instant::now();

        Ok(Self {
            id,
            browser,
            created_at: now,
            last_used: now,
            page_count: 0,
            is_healthy: true,
            stealth_mode,
        })
    }

    /// Create new page with stealth mode
    async fn new_page(&mut self, url: &str) -> Result<Page> {
        let page = self
            .browser
            .new_page(url)
            .await
            .context("Failed to create page")?;

        // Apply stealth mode if enabled
        if let Some(stealth) = &self.stealth_mode {
            stealth.apply(&page).await?;
        }

        self.page_count += 1;
        self.last_used = Instant::now();

        Ok(page)
    }

    /// Check if browser is idle
    fn is_idle(&self, max_idle_time: Duration) -> bool {
        self.last_used.elapsed() > max_idle_time
    }

    /// Check if browser is healthy
    async fn health_check(&mut self) -> bool {
        // Try to get browser version as health check
        match self.browser.version().await {
            Ok(_) => {
                self.is_healthy = true;
                true
            }
            Err(e) => {
                warn!("Browser {} health check failed: {}", self.id, e);
                self.is_healthy = false;
                false
            }
        }
    }

    /// Close browser
    async fn close(self) -> Result<()> {
        info!("Closing browser instance: {}", self.id);
        self.browser.close().await?;
        Ok(())
    }
}

/// Browser pool manager
pub struct BrowserPool {
    config: PoolConfig,
    instances: Arc<RwLock<HashMap<String, BrowserInstance>>>,
    semaphore: Arc<Semaphore>,
    user_agent_manager: Arc<RwLock<UserAgentManager>>,
    stealth_mode: Option<Arc<StealthMode>>,
    next_id: Arc<RwLock<usize>>,
}

impl BrowserPool {
    /// Create new browser pool
    pub async fn new(config: PoolConfig) -> Result<Self> {
        let stealth_mode = if config.enable_stealth {
            Some(Arc::new(StealthMode::new(config.stealth_config.clone())))
        } else {
            None
        };

        let pool = Self {
            config: config.clone(),
            instances: Arc::new(RwLock::new(HashMap::new())),
            semaphore: Arc::new(Semaphore::new(config.max_instances)),
            user_agent_manager: Arc::new(RwLock::new(UserAgentManager::new())),
            stealth_mode,
            next_id: Arc::new(RwLock::new(0)),
        };

        // Create initial instances
        pool.scale_to(config.initial_instances).await?;

        // Start background tasks
        pool.start_health_checker();
        pool.start_idle_cleanup();

        info!(
            "Browser pool initialized with {} instances",
            config.initial_instances
        );

        Ok(pool)
    }

    /// Acquire a browser instance from the pool
    pub async fn acquire(&self) -> Result<BrowserGuard> {
        // Wait for available slot
        let permit = self.semaphore.acquire().await?;

        // Try to get existing healthy instance
        let instance_id = {
            let instances = self.instances.read().await;
            instances
                .iter()
                .find(|(_, inst)| inst.is_healthy)
                .map(|(id, _)| id.clone())
        };

        // If no healthy instance, create new one
        let instance_id = if let Some(id) = instance_id {
            id
        } else {
            self.create_instance().await?
        };

        Ok(BrowserGuard {
            pool: self.clone(),
            instance_id: Some(instance_id),
            _permit: permit,
        })
    }

    /// Create new browser instance
    async fn create_instance(&self) -> Result<String> {
        let id = {
            let mut next_id = self.next_id.write().await;
            let id = format!("browser-{}", *next_id);
            *next_id += 1;
            id
        };

        info!("Creating new browser instance: {}", id);

        let instance =
            BrowserInstance::new(id.clone(), &self.config, self.stealth_mode.clone()).await?;

        self.instances.write().await.insert(id.clone(), instance);

        Ok(id)
    }

    /// Get page from instance
    async fn get_page(&self, instance_id: &str, url: &str) -> Result<Page> {
        let mut instances = self.instances.write().await;
        let instance = instances
            .get_mut(instance_id)
            .context("Instance not found")?;

        instance.new_page(url).await
    }

    /// Scale pool to target size
    async fn scale_to(&self, target: usize) -> Result<()> {
        let current = self.instances.read().await.len();

        if target > current {
            // Scale up
            let to_add = target - current;
            info!("Scaling up browser pool by {} instances", to_add);

            for _ in 0..to_add {
                self.create_instance().await?;
            }
        } else if target < current {
            // Scale down
            let to_remove = current - target;
            info!("Scaling down browser pool by {} instances", to_remove);

            self.remove_idle_instances(to_remove).await?;
        }

        Ok(())
    }

    /// Remove idle instances
    async fn remove_idle_instances(&self, count: usize) -> Result<()> {
        let mut instances = self.instances.write().await;

        // Find idle instances
        let mut idle_ids: Vec<String> = instances
            .iter()
            .filter(|(_, inst)| inst.is_idle(self.config.max_idle_time))
            .take(count)
            .map(|(id, _)| id.clone())
            .collect();

        // Remove them
        for id in idle_ids.drain(..) {
            if let Some(instance) = instances.remove(&id) {
                tokio::spawn(async move {
                    if let Err(e) = instance.close().await {
                        error!("Failed to close browser {}: {}", id, e);
                    }
                });
            }
        }

        Ok(())
    }

    /// Start health checker background task
    fn start_health_checker(&self) {
        let instances = self.instances.clone();
        let interval = self.config.health_check_interval;

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;

                let mut instances = instances.write().await;
                for (id, instance) in instances.iter_mut() {
                    if !instance.health_check().await {
                        warn!("Browser {} is unhealthy", id);
                    }
                }
            }
        });
    }

    /// Start idle cleanup background task
    fn start_idle_cleanup(&self) {
        let pool = self.clone();
        let min_instances = self.config.min_instances;

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;

                let current_count = pool.instances.read().await.len();

                if current_count > min_instances {
                    let to_remove = current_count - min_instances;
                    if let Err(e) = pool.remove_idle_instances(to_remove).await {
                        error!("Failed to remove idle instances: {}", e);
                    }
                }
            }
        });
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        let instances = self.instances.read().await;

        let total_instances = instances.len();
        let healthy_instances = instances.values().filter(|i| i.is_healthy).count();
        let total_pages: usize = instances.values().map(|i| i.page_count).sum();
        let idle_instances = instances
            .values()
            .filter(|i| i.is_idle(self.config.max_idle_time))
            .count();

        PoolStats {
            total_instances,
            healthy_instances,
            idle_instances,
            total_pages,
            available_slots: self.semaphore.available_permits(),
        }
    }

    /// Close all browser instances
    pub async fn close_all(&self) -> Result<()> {
        info!("Closing all browser instances...");

        let mut instances = self.instances.write().await;
        let instance_ids: Vec<String> = instances.keys().cloned().collect();

        for id in instance_ids {
            if let Some(instance) = instances.remove(&id) {
                tokio::spawn(async move {
                    if let Err(e) = instance.close().await {
                        error!("Failed to close browser {}: {}", id, e);
                    }
                });
            }
        }

        Ok(())
    }
}

impl Clone for BrowserPool {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            instances: self.instances.clone(),
            semaphore: self.semaphore.clone(),
            user_agent_manager: self.user_agent_manager.clone(),
            stealth_mode: self.stealth_mode.clone(),
            next_id: self.next_id.clone(),
        }
    }
}

/// Browser guard (RAII pattern)
///
/// Automatically returns browser to pool when dropped
pub struct BrowserGuard {
    pool: BrowserPool,
    instance_id: Option<String>,
    _permit: tokio::sync::SemaphorePermit<'static>,
}

impl BrowserGuard {
    /// Get new page from this browser
    pub async fn new_page(&self, url: &str) -> Result<Page> {
        let instance_id = self.instance_id.as_ref().context("No instance ID")?;

        self.pool.get_page(instance_id, url).await
    }

    /// Get instance ID
    pub fn instance_id(&self) -> Option<&str> {
        self.instance_id.as_deref()
    }
}

impl Drop for BrowserGuard {
    fn drop(&mut self) {
        // Permit is automatically released when dropped
        if let Some(id) = &self.instance_id {
            debug!("Returning browser {} to pool", id);
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_instances: usize,
    pub healthy_instances: usize,
    pub idle_instances: usize,
    pub total_pages: usize,
    pub available_slots: usize,
}

impl PoolStats {
    pub fn utilization(&self) -> f32 {
        if self.total_instances == 0 {
            return 0.0;
        }
        let used = self.total_instances - self.idle_instances;
        used as f32 / self.total_instances as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pool_creation() {
        let config = PoolConfig {
            initial_instances: 1,
            min_instances: 1,
            max_instances: 5,
            ..Default::default()
        };

        let pool = BrowserPool::new(config).await.unwrap();
        let stats = pool.stats().await;

        assert_eq!(stats.total_instances, 1);
    }

    #[tokio::test]
    async fn test_acquire_browser() {
        let config = PoolConfig {
            initial_instances: 2,
            ..Default::default()
        };

        let pool = BrowserPool::new(config).await.unwrap();
        let guard = pool.acquire().await.unwrap();

        assert!(guard.instance_id().is_some());
    }

    #[tokio::test]
    async fn test_pool_stats() {
        let config = PoolConfig {
            initial_instances: 3,
            ..Default::default()
        };

        let pool = BrowserPool::new(config).await.unwrap();
        let stats = pool.stats().await;

        assert_eq!(stats.total_instances, 3);
        assert_eq!(stats.healthy_instances, 3);
    }
}
