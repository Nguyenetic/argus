//! Storage layer using NexusQL brain-inspired database

use anyhow::Result;
use nexusql::{Database, ExecuteParams};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize)]
pub struct ScrapedPage {
    pub id: String,
    pub url: String,
    pub title: Option<String>,
    pub content: String,
    pub links: Vec<String>,
    pub scraped_at: String,
    pub content_length: usize,
}

pub struct NexusStorage {
    db: Database,
}

impl NexusStorage {
    pub async fn new(db_path: &str) -> Result<Self> {
        let db = Database::open(db_path).await?;

        // Create schema for scraped pages
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS pages (
                id TEXT PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                links TEXT, -- JSON array
                scraped_at TEXT NOT NULL,
                content_length INTEGER NOT NULL,
                -- Vector embedding for semantic search (using HNSW)
                content_vector VECTOR(384)
            )
            "#,
            ExecuteParams::default(),
        ).await?;

        // Create HNSW index for vector search
        db.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_pages_vector
            ON pages USING hnsw(content_vector)
            WITH (m = 16, ef_construction = 200)
            "#,
            ExecuteParams::default(),
        ).await?;

        // Create full-text index
        db.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_pages_content
            ON pages USING gin(to_tsvector('english', content))
            "#,
            ExecuteParams::default(),
        ).await?;

        Ok(Self { db })
    }

    pub async fn save_page(&self, page: &ScrapedPage) -> Result<()> {
        let links_json = serde_json::to_string(&page.links)?;

        self.db.execute(
            r#"
            INSERT INTO pages (id, url, title, content, links, scraped_at, content_length)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (url) DO UPDATE
            SET title = $3, content = $4, links = $5, scraped_at = $6, content_length = $7
            "#,
            ExecuteParams::new(vec![
                page.id.clone().into(),
                page.url.clone().into(),
                page.title.clone().into(),
                page.content.clone().into(),
                links_json.into(),
                page.scraped_at.clone().into(),
                (page.content_length as i64).into(),
            ]),
        ).await?;

        Ok(())
    }

    pub async fn get_page_by_url(&self, url: &str) -> Result<Option<ScrapedPage>> {
        let result = self.db.query(
            "SELECT id, url, title, content, links, scraped_at, content_length FROM pages WHERE url = $1",
            ExecuteParams::new(vec![url.to_string().into()]),
        ).await?;

        if result.rows.is_empty() {
            return Ok(None);
        }

        let row = &result.rows[0];
        let links: Vec<String> = serde_json::from_str(row[4].as_str().unwrap_or("[]"))?;

        Ok(Some(ScrapedPage {
            id: row[0].as_str().unwrap_or_default().to_string(),
            url: row[1].as_str().unwrap_or_default().to_string(),
            title: row[2].as_str().map(|s| s.to_string()),
            content: row[3].as_str().unwrap_or_default().to_string(),
            links,
            scraped_at: row[5].as_str().unwrap_or_default().to_string(),
            content_length: row[6].as_i64().unwrap_or(0) as usize,
        }))
    }

    pub async fn list_all(&self) -> Result<Vec<ScrapedPage>> {
        let result = self.db.query(
            "SELECT id, url, title, content, links, scraped_at, content_length FROM pages ORDER BY scraped_at DESC",
            ExecuteParams::default(),
        ).await?;

        let mut pages = Vec::new();
        for row in result.rows {
            let links: Vec<String> = serde_json::from_str(row[4].as_str().unwrap_or("[]"))?;

            pages.push(ScrapedPage {
                id: row[0].as_str().unwrap_or_default().to_string(),
                url: row[1].as_str().unwrap_or_default().to_string(),
                title: row[2].as_str().map(|s| s.to_string()),
                content: row[3].as_str().unwrap_or_default().to_string(),
                links,
                scraped_at: row[5].as_str().unwrap_or_default().to_string(),
                content_length: row[6].as_i64().unwrap_or(0) as usize,
            });
        }

        Ok(pages)
    }

    pub async fn get_stats(&self) -> Result<(usize, usize, usize)> {
        let result = self.db.query(
            "SELECT COUNT(*), SUM(content_length), SUM(json_array_length(links)) FROM pages",
            ExecuteParams::default(),
        ).await?;

        if result.rows.is_empty() {
            return Ok((0, 0, 0));
        }

        let row = &result.rows[0];
        Ok((
            row[0].as_i64().unwrap_or(0) as usize,
            row[1].as_i64().unwrap_or(0) as usize,
            row[2].as_i64().unwrap_or(0) as usize,
        ))
    }

    /// Semantic search using HDC or ColBERT
    pub async fn semantic_search(&self, query: &str, limit: usize) -> Result<Vec<ScrapedPage>> {
        // TODO: Use NexusQL's HDC or ColBERT for semantic search
        // For now, use full-text search
        let result = self.db.query(
            r#"
            SELECT id, url, title, content, links, scraped_at, content_length
            FROM pages
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
            ORDER BY ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) DESC
            LIMIT $2
            "#,
            ExecuteParams::new(vec![query.to_string().into(), (limit as i64).into()]),
        ).await?;

        let mut pages = Vec::new();
        for row in result.rows {
            let links: Vec<String> = serde_json::from_str(row[4].as_str().unwrap_or("[]"))?;

            pages.push(ScrapedPage {
                id: row[0].as_str().unwrap_or_default().to_string(),
                url: row[1].as_str().unwrap_or_default().to_string(),
                title: row[2].as_str().map(|s| s.to_string()),
                content: row[3].as_str().unwrap_or_default().to_string(),
                links,
                scraped_at: row[5].as_str().unwrap_or_default().to_string(),
                content_length: row[6].as_i64().unwrap_or(0) as usize,
            });
        }

        Ok(pages)
    }
}
