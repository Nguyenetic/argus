//! Database operations using SQLx and PostgreSQL

use argus_core::{Error, Result, Page};
use sqlx::{PgPool, postgres::PgPoolOptions};
use tracing::info;

pub struct Database {
    pool: PgPool,
}

impl Database {
    pub async fn connect(database_url: &str) -> Result<Self> {
        info!("Connecting to database: {}", database_url);

        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .map_err(|e| Error::DatabaseError(format!("Connection failed: {}", e)))?;

        info!("Database connected successfully");

        Ok(Self { pool })
    }

    pub async fn save_page(&self, page: &Page) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO pages (id, url, title, content, html, status, scraped_at, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (url) DO UPDATE
            SET title = $3, content = $4, html = $5, status = $6, scraped_at = $7, metadata = $8
            "#,
            page.id,
            page.url,
            page.title,
            page.content,
            page.html,
            serde_json::to_value(&page.status).unwrap(),
            page.scraped_at,
            page.metadata
        )
        .execute(&self.pool)
        .await
        .map_err(|e| Error::DatabaseError(format!("Failed to save page: {}", e)))?;

        Ok(())
    }

    pub async fn get_page_by_url(&self, url: &str) -> Result<Option<Page>> {
        let result = sqlx::query_as!(
            PageRow,
            r#"
            SELECT id, url, title, content, html, status as "status: serde_json::Value",
                   scraped_at, metadata
            FROM pages
            WHERE url = $1
            "#,
            url
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Error::DatabaseError(format!("Query failed: {}", e)))?;

        Ok(result.map(|row| row.into_page()))
    }
}

#[derive(sqlx::FromRow)]
struct PageRow {
    id: uuid::Uuid,
    url: String,
    title: Option<String>,
    content: String,
    html: Option<String>,
    status: serde_json::Value,
    scraped_at: chrono::DateTime<chrono::Utc>,
    metadata: Option<serde_json::Value>,
}

impl PageRow {
    fn into_page(self) -> Page {
        Page {
            id: self.id,
            url: self.url,
            title: self.title,
            content: self.content,
            html: self.html,
            status: serde_json::from_value(self.status).unwrap_or(argus_core::PageStatus::Pending),
            scraped_at: self.scraped_at,
            metadata: self.metadata,
        }
    }
}
