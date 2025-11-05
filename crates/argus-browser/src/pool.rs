//! Browser pool manager for efficient parallel scraping

use crate::ChromeBrowser;
use argus_core::Result;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, info};

/// Browser pool for managing multiple browser instances
pub struct BrowserPool {
    browsers: Arc<Mutex<Vec<Arc<ChromeBrowser>>>>,
    semaphore: Arc<Semaphore>,
    max_size: usize,
}

impl BrowserPool {
    /// Create a new browser pool with the specified size
    pub async fn new(size: usize) -> Result<Self> {
        info!("Creating browser pool with {} instances", size);

        let mut browsers = Vec::with_capacity(size);

        // Pre-create browser instances
        for i in 0..size {
            debug!("Initializing browser {} of {}", i + 1, size);
            let browser = ChromeBrowser::new(true).await?;
            browsers.push(Arc::new(browser));
        }

        info!("Browser pool initialized with {} browsers", size);

        Ok(Self {
            browsers: Arc::new(Mutex::new(browsers)),
            semaphore: Arc::new(Semaphore::new(size)),
            max_size: size,
        })
    }

    /// Acquire a browser from the pool
    /// Blocks if all browsers are in use
    pub async fn acquire(&self) -> Result<BrowserGuard> {
        // Acquire semaphore permit
        let permit = self.semaphore.clone().acquire_owned().await.map_err(|e| {
            argus_core::Error::BrowserError(format!("Failed to acquire browser: {}", e))
        })?;

        // Get a browser from the pool
        let mut browsers = self.browsers.lock().await;
        let browser = browsers
            .pop()
            .ok_or_else(|| argus_core::Error::BrowserError("No browsers available".to_string()))?;

        debug!("Browser acquired from pool ({} remaining)", browsers.len());

        Ok(BrowserGuard {
            browser: Some(browser),
            pool: self.browsers.clone(),
            _permit: permit,
        })
    }

    /// Get the pool size
    pub fn size(&self) -> usize {
        self.max_size
    }

    /// Get the number of available browsers
    pub async fn available(&self) -> usize {
        self.browsers.lock().await.len()
    }
}

/// RAII guard for browser - automatically returns browser to pool when dropped
pub struct BrowserGuard {
    browser: Option<Arc<ChromeBrowser>>,
    pool: Arc<Mutex<Vec<Arc<ChromeBrowser>>>>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl BrowserGuard {
    /// Get a reference to the browser
    pub fn browser(&self) -> &ChromeBrowser {
        self.browser.as_ref().unwrap()
    }
}

impl Drop for BrowserGuard {
    fn drop(&mut self) {
        if let Some(browser) = self.browser.take() {
            let pool = self.pool.clone();
            tokio::spawn(async move {
                let mut browsers = pool.lock().await;
                browsers.push(browser);
                debug!("Browser returned to pool ({} available)", browsers.len());
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires Chrome to be installed
    async fn test_browser_pool_creation() {
        let pool = BrowserPool::new(2).await;
        assert!(pool.is_ok());

        let pool = pool.unwrap();
        assert_eq!(pool.size(), 2);
        assert_eq!(pool.available().await, 2);
    }

    #[tokio::test]
    #[ignore]
    async fn test_browser_acquire_and_release() {
        let pool = BrowserPool::new(1).await.unwrap();

        assert_eq!(pool.available().await, 1);

        {
            let _guard = pool.acquire().await.unwrap();
            // Browser acquired
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            assert_eq!(pool.available().await, 0);
        }

        // Browser should be returned after guard drops
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        assert_eq!(pool.available().await, 1);
    }
}
