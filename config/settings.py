"""
Global configuration settings for the web scraper
Loads from environment variables with sensible defaults
"""
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # ============================================
    # DATABASE CONFIGURATION
    # ============================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://scraper:scraper123@localhost:5432/webscraper",
        description="PostgreSQL connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, description="Connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=40, description="Max overflow connections")

    # Vector configuration
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    EMBEDDING_DIMENSION: int = Field(default=384, description="Embedding vector dimension")

    # ============================================
    # REDIS CONFIGURATION
    # ============================================
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    REDIS_CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")
    REDIS_MAX_CONNECTIONS: int = Field(default=50, description="Max Redis connections")

    # ============================================
    # CELERY CONFIGURATION
    # ============================================
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend"
    )
    CELERY_WORKER_CONCURRENCY: int = Field(default=4, description="Worker concurrency")
    CELERY_TASK_TIMEOUT: int = Field(default=300, description="Task timeout in seconds")

    # ============================================
    # S3/MINIO CONFIGURATION
    # ============================================
    S3_ENDPOINT_URL: str = Field(
        default="http://localhost:9000",
        description="S3/MinIO endpoint"
    )
    S3_ACCESS_KEY: str = Field(default="minioadmin", description="S3 access key")
    S3_SECRET_KEY: str = Field(default="minioadmin", description="S3 secret key")
    S3_BUCKET_NAME: str = Field(default="web-scraper", description="S3 bucket name")
    S3_REGION: str = Field(default="us-east-1", description="S3 region")
    S3_USE_SSL: bool = Field(default=False, description="Use SSL for S3")

    # ============================================
    # SCRAPER CONFIGURATION
    # ============================================
    # Proxy configuration (comma-separated)
    PROXY_TIER_1: Optional[str] = Field(default=None, description="Tier 1 proxies (free/cheap)")
    PROXY_TIER_2: Optional[str] = Field(default=None, description="Tier 2 proxies (mid-tier)")
    PROXY_TIER_3: Optional[str] = Field(default=None, description="Tier 3 proxies (premium)")

    # Rate limiting
    DEFAULT_DELAY_SECONDS: int = Field(default=5, description="Default delay between requests")
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, description="Max concurrent requests")
    REQUEST_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")

    # User agents
    USER_AGENT_ROTATION: bool = Field(default=True, description="Enable UA rotation")

    # ============================================
    # SCRAPER STRATEGIES
    # ============================================
    ENABLE_CRAWLEE: bool = Field(default=True, description="Enable Crawlee scraper")
    ENABLE_SELENIUM: bool = Field(default=True, description="Enable SeleniumBase scraper")
    ENABLE_SCRAPLING: bool = Field(default=True, description="Enable Scrapling scraper")

    # SeleniumBase
    SELENIUM_HEADLESS: bool = Field(default=True, description="Run Selenium in headless mode")
    SELENIUM_SOLVE_CAPTCHA: bool = Field(default=True, description="Auto-solve CAPTCHAs")

    # Scrapling
    SCRAPLING_STEALTH: bool = Field(default=True, description="Enable Scrapling stealth mode")
    SCRAPLING_CLOUDFLARE_SOLVE: bool = Field(
        default=True,
        description="Auto-solve Cloudflare"
    )

    # ============================================
    # KNOWLEDGE GRAPH CONFIGURATION
    # ============================================
    ENABLE_ENTITY_EXTRACTION: bool = Field(
        default=True,
        description="Enable entity extraction"
    )
    ENTITY_EXTRACTION_MODEL: str = Field(
        default="en_core_web_sm",
        description="spaCy model for NER"
    )
    MIN_ENTITY_CONFIDENCE: float = Field(
        default=0.7,
        description="Min confidence for entities"
    )

    # ============================================
    # MONITORING & LOGGING
    # ============================================
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="logs/scraper.log", description="Log file path")
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN")

    # Prometheus
    PROMETHEUS_PORT: int = Field(default=9090, description="Prometheus port")
    METRICS_ENABLED: bool = Field(default=True, description="Enable metrics")

    # ============================================
    # API CONFIGURATION
    # ============================================
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    API_RELOAD: bool = Field(default=True, description="Auto-reload API")
    API_WORKERS: int = Field(default=4, description="API workers")

    # Authentication
    JWT_SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        description="JWT secret key"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="Access token expiration"
    )

    # ============================================
    # FEATURE FLAGS
    # ============================================
    ENABLE_HYBRID_SEARCH: bool = Field(default=True, description="Enable hybrid search")
    ENABLE_KNOWLEDGE_GRAPH: bool = Field(default=True, description="Enable knowledge graph")
    ENABLE_ADAPTIVE_SELECTORS: bool = Field(
        default=True,
        description="Enable adaptive selectors"
    )
    ENABLE_AUTO_RETRY: bool = Field(default=True, description="Enable auto-retry")
    MAX_RETRY_ATTEMPTS: int = Field(default=3, description="Max retry attempts")

    # ============================================
    # VALIDATORS
    # ============================================
    @validator("PROXY_TIER_1", "PROXY_TIER_2", "PROXY_TIER_3", pre=True)
    def parse_proxy_list(cls, v):
        """Parse comma-separated proxy list"""
        if v is None or v == "":
            return []
        return [proxy.strip() for proxy in v.split(",")]

    @validator("EMBEDDING_DIMENSION")
    def validate_embedding_dimension(cls, v):
        """Ensure embedding dimension is valid"""
        if v <= 0 or v > 2000:
            raise ValueError("Embedding dimension must be between 1 and 2000")
        return v

    @validator("MIN_ENTITY_CONFIDENCE")
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Helper functions
def get_proxy_tiers() -> List[List[str]]:
    """Get proxy configuration in tiered format"""
    tiers = []
    if settings.PROXY_TIER_1:
        tiers.append(settings.PROXY_TIER_1)
    if settings.PROXY_TIER_2:
        tiers.append(settings.PROXY_TIER_2)
    if settings.PROXY_TIER_3:
        tiers.append(settings.PROXY_TIER_3)
    return tiers


def is_production() -> bool:
    """Check if running in production"""
    return os.getenv("ENVIRONMENT", "development") == "production"
