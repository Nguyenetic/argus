//! Redis caching layer

use argus_core::{Error, Result};
use redis::{Client, AsyncCommands};
use tracing::info;

pub struct Cache {
    client: Client,
}

impl Cache {
    pub async fn connect(redis_url: &str) -> Result<Self> {
        info!("Connecting to Redis: {}", redis_url);

        let client = Client::open(redis_url)
            .map_err(|e| Error::DatabaseError(format!("Redis connection failed: {}", e)))?;

        info!("Redis connected successfully");

        Ok(Self { client })
    }

    pub async fn get(&self, key: &str) -> Result<Option<String>> {
        let mut conn = self.client.get_async_connection()
            .await
            .map_err(|e| Error::DatabaseError(format!("Failed to get connection: {}", e)))?;

        let value: Option<String> = conn.get(key)
            .await
            .map_err(|e| Error::DatabaseError(format!("Cache get failed: {}", e)))?;

        Ok(value)
    }

    pub async fn set(&self, key: &str, value: &str, ttl_seconds: usize) -> Result<()> {
        let mut conn = self.client.get_async_connection()
            .await
            .map_err(|e| Error::DatabaseError(format!("Failed to get connection: {}", e)))?;

        conn.set_ex(key, value, ttl_seconds)
            .await
            .map_err(|e| Error::DatabaseError(format!("Cache set failed: {}", e)))?;

        Ok(())
    }
}
