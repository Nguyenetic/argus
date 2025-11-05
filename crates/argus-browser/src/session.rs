/// Session persistence for browser automation
///
/// Manages browser sessions with:
/// - Cookie persistence
/// - Local storage persistence
/// - Session restoration
/// - Authentication state
/// - Multi-session management
use anyhow::{Context, Result};
use chromiumoxide::cdp::browser_protocol::network::{Cookie, CookieParam};
use chromiumoxide::cdp::browser_protocol::storage::{
    ClearCookiesParams, GetCookiesParams, SetCookiesParams,
};
use chromiumoxide::Page;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

/// Session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub id: String,
    pub cookies: Vec<SerializableCookie>,
    pub local_storage: HashMap<String, String>,
    pub session_storage: HashMap<String, String>,
    pub user_agent: String,
    pub viewport: (u32, u32),
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

/// Serializable cookie (simpler than Chrome CDP Cookie)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCookie {
    pub name: String,
    pub value: String,
    pub domain: String,
    pub path: String,
    pub expires: Option<f64>,
    pub http_only: bool,
    pub secure: bool,
    pub same_site: Option<String>,
}

impl From<Cookie> for SerializableCookie {
    fn from(cookie: Cookie) -> Self {
        Self {
            name: cookie.name,
            value: cookie.value,
            domain: cookie.domain,
            path: cookie.path,
            expires: cookie.expires,
            http_only: cookie.http_only.unwrap_or(false),
            secure: cookie.secure.unwrap_or(false),
            same_site: cookie.same_site.map(|s| format!("{:?}", s)),
        }
    }
}

impl From<SerializableCookie> for CookieParam {
    fn from(cookie: SerializableCookie) -> Self {
        CookieParam::builder()
            .name(cookie.name)
            .value(cookie.value)
            .domain(cookie.domain)
            .path(cookie.path)
            .expires(cookie.expires)
            .http_only(cookie.http_only)
            .secure(cookie.secure)
            .build()
            .unwrap()
    }
}

/// Session manager
pub struct SessionManager {
    storage_dir: PathBuf,
}

impl SessionManager {
    /// Create new session manager
    pub fn new<P: AsRef<Path>>(storage_dir: P) -> Result<Self> {
        let storage_dir = storage_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&storage_dir)?;

        Ok(Self { storage_dir })
    }

    /// Save session data
    pub async fn save_session(&self, session: &SessionData) -> Result<()> {
        let path = self.session_path(&session.id);

        let json = serde_json::to_string_pretty(session).context("Failed to serialize session")?;

        fs::write(&path, json)
            .await
            .context("Failed to write session file")?;

        info!("Session saved: {}", session.id);
        Ok(())
    }

    /// Load session data
    pub async fn load_session(&self, session_id: &str) -> Result<SessionData> {
        let path = self.session_path(session_id);

        let json = fs::read_to_string(&path)
            .await
            .context("Failed to read session file")?;

        let session: SessionData =
            serde_json::from_str(&json).context("Failed to deserialize session")?;

        info!("Session loaded: {}", session_id);
        Ok(session)
    }

    /// Delete session
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        let path = self.session_path(session_id);

        fs::remove_file(&path)
            .await
            .context("Failed to delete session file")?;

        info!("Session deleted: {}", session_id);
        Ok(())
    }

    /// List all sessions
    pub async fn list_sessions(&self) -> Result<Vec<String>> {
        let mut sessions = Vec::new();

        let mut entries = fs::read_dir(&self.storage_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".json") {
                    sessions.push(name.trim_end_matches(".json").to_string());
                }
            }
        }

        Ok(sessions)
    }

    /// Get session file path
    fn session_path(&self, session_id: &str) -> PathBuf {
        self.storage_dir.join(format!("{}.json", session_id))
    }
}

/// Session extractor - extracts session data from page
pub struct SessionExtractor;

impl SessionExtractor {
    /// Extract complete session data from page
    pub async fn extract(page: &Page) -> Result<SessionData> {
        let id = uuid::Uuid::new_v4().to_string();

        // Extract cookies
        let cookies = Self::extract_cookies(page).await?;

        // Extract local storage
        let local_storage = Self::extract_local_storage(page).await?;

        // Extract session storage
        let session_storage = Self::extract_session_storage(page).await?;

        // Get user agent
        let user_agent = Self::get_user_agent(page).await?;

        // Get viewport
        let viewport = Self::get_viewport(page).await?;

        let now = chrono::Utc::now();

        Ok(SessionData {
            id,
            cookies,
            local_storage,
            session_storage,
            user_agent,
            viewport,
            created_at: now,
            last_used: now,
            metadata: HashMap::new(),
        })
    }

    /// Extract cookies from page
    async fn extract_cookies(page: &Page) -> Result<Vec<SerializableCookie>> {
        let result = page.execute(GetCookiesParams::default()).await?;

        let cookies = result
            .result
            .cookies
            .into_iter()
            .map(SerializableCookie::from)
            .collect();

        debug!("Extracted {} cookies", cookies.len());
        Ok(cookies)
    }

    /// Extract local storage
    async fn extract_local_storage(page: &Page) -> Result<HashMap<String, String>> {
        let js = r#"
            (() => {
                const storage = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    storage[key] = localStorage.getItem(key);
                }
                return JSON.stringify(storage);
            })()
        "#;

        let result = page.evaluate(js).await?;
        let json: String = result.into_value()?;
        let storage: HashMap<String, String> = serde_json::from_str(&json)?;

        debug!("Extracted {} localStorage items", storage.len());
        Ok(storage)
    }

    /// Extract session storage
    async fn extract_session_storage(page: &Page) -> Result<HashMap<String, String>> {
        let js = r#"
            (() => {
                const storage = {};
                for (let i = 0; i < sessionStorage.length; i++) {
                    const key = sessionStorage.key(i);
                    storage[key] = sessionStorage.getItem(key);
                }
                return JSON.stringify(storage);
            })()
        "#;

        let result = page.evaluate(js).await?;
        let json: String = result.into_value()?;
        let storage: HashMap<String, String> = serde_json::from_str(&json)?;

        debug!("Extracted {} sessionStorage items", storage.len());
        Ok(storage)
    }

    /// Get user agent
    async fn get_user_agent(page: &Page) -> Result<String> {
        let js = "navigator.userAgent";
        let result = page.evaluate(js).await?;
        Ok(result.into_value()?)
    }

    /// Get viewport size
    async fn get_viewport(page: &Page) -> Result<(u32, u32)> {
        let js = "JSON.stringify([window.innerWidth, window.innerHeight])";
        let result = page.evaluate(js).await?;
        let json: String = result.into_value()?;
        let dimensions: Vec<u32> = serde_json::from_str(&json)?;

        Ok((dimensions[0], dimensions[1]))
    }
}

/// Session restorer - restores session data to page
pub struct SessionRestorer;

impl SessionRestorer {
    /// Restore complete session to page
    pub async fn restore(page: &Page, session: &SessionData) -> Result<()> {
        info!("Restoring session: {}", session.id);

        // Restore cookies
        Self::restore_cookies(page, &session.cookies).await?;

        // Restore local storage
        Self::restore_local_storage(page, &session.local_storage).await?;

        // Restore session storage
        Self::restore_session_storage(page, &session.session_storage).await?;

        info!("Session restored successfully");
        Ok(())
    }

    /// Restore cookies to page
    async fn restore_cookies(page: &Page, cookies: &[SerializableCookie]) -> Result<()> {
        let cookie_params: Vec<CookieParam> =
            cookies.iter().cloned().map(CookieParam::from).collect();

        if !cookie_params.is_empty() {
            page.execute(SetCookiesParams {
                cookies: cookie_params,
            })
            .await?;

            debug!("Restored {} cookies", cookies.len());
        }

        Ok(())
    }

    /// Restore local storage
    async fn restore_local_storage(page: &Page, storage: &HashMap<String, String>) -> Result<()> {
        if storage.is_empty() {
            return Ok(());
        }

        let json = serde_json::to_string(storage)?;

        let js = format!(
            r#"
            (() => {{
                const storage = {};
                for (const [key, value] of Object.entries(storage)) {{
                    localStorage.setItem(key, value);
                }}
            }})()
        "#,
            json
        );

        page.evaluate(&js).await?;

        debug!("Restored {} localStorage items", storage.len());
        Ok(())
    }

    /// Restore session storage
    async fn restore_session_storage(page: &Page, storage: &HashMap<String, String>) -> Result<()> {
        if storage.is_empty() {
            return Ok(());
        }

        let json = serde_json::to_string(storage)?;

        let js = format!(
            r#"
            (() => {{
                const storage = {};
                for (const [key, value] of Object.entries(storage)) {{
                    sessionStorage.setItem(key, value);
                }}
            }})()
        "#,
            json
        );

        page.evaluate(&js).await?;

        debug!("Restored {} sessionStorage items", storage.len());
        Ok(())
    }

    /// Clear all session data
    pub async fn clear(page: &Page) -> Result<()> {
        // Clear cookies
        page.execute(ClearCookiesParams::default()).await?;

        // Clear storage
        let js = r#"
            localStorage.clear();
            sessionStorage.clear();
        "#;
        page.evaluate(js).await?;

        debug!("Session data cleared");
        Ok(())
    }
}

/// Session lifecycle manager
pub struct SessionLifecycle {
    manager: SessionManager,
}

impl SessionLifecycle {
    /// Create new session lifecycle manager
    pub fn new(storage_dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            manager: SessionManager::new(storage_dir)?,
        })
    }

    /// Create and save new session from page
    pub async fn create_session(&self, page: &Page) -> Result<String> {
        let session = SessionExtractor::extract(page).await?;
        let session_id = session.id.clone();

        self.manager.save_session(&session).await?;

        Ok(session_id)
    }

    /// Resume existing session on page
    pub async fn resume_session(&self, page: &Page, session_id: &str) -> Result<()> {
        let mut session = self.manager.load_session(session_id).await?;

        // Update last used
        session.last_used = chrono::Utc::now();

        // Restore to page
        SessionRestorer::restore(page, &session).await?;

        // Save updated session
        self.manager.save_session(&session).await?;

        Ok(())
    }

    /// Update existing session
    pub async fn update_session(&self, page: &Page, session_id: &str) -> Result<()> {
        let mut session = SessionExtractor::extract(page).await?;

        // Keep original ID and timestamps
        let original = self.manager.load_session(session_id).await?;
        session.id = original.id;
        session.created_at = original.created_at;
        session.last_used = chrono::Utc::now();
        session.metadata = original.metadata;

        self.manager.save_session(&session).await?;

        Ok(())
    }

    /// Delete session
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        self.manager.delete_session(session_id).await
    }

    /// List all sessions
    pub async fn list_sessions(&self) -> Result<Vec<String>> {
        self.manager.list_sessions().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_session_manager() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SessionManager::new(temp_dir.path()).unwrap();

        let session = SessionData {
            id: "test-session".to_string(),
            cookies: vec![],
            local_storage: HashMap::new(),
            session_storage: HashMap::new(),
            user_agent: "Test".to_string(),
            viewport: (1920, 1080),
            created_at: chrono::Utc::now(),
            last_used: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        // Save
        manager.save_session(&session).await.unwrap();

        // Load
        let loaded = manager.load_session("test-session").await.unwrap();
        assert_eq!(loaded.id, "test-session");

        // List
        let sessions = manager.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 1);

        // Delete
        manager.delete_session("test-session").await.unwrap();
        let sessions = manager.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 0);
    }
}
