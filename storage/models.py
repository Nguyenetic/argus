"""
SQLAlchemy models with pgvector support for web scraping
Includes tables for scraped data, knowledge graph, jobs, and monitoring
"""
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, TIMESTAMP,
    ForeignKey, JSON, ARRAY, UniqueConstraint, Index, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

Base = declarative_base()


# ============================================
# ENUMS
# ============================================

class PageStatus(str, Enum):
    """Status of scraped page"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, Enum):
    """Status of scraping job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScraperType(str, Enum):
    """Type of scraper used"""
    CRAWLEE = "crawlee"
    SELENIUM = "selenium"
    SCRAPLING = "scrapling"
    HYBRID = "hybrid"


class EntityType(str, Enum):
    """Types of entities in knowledge graph"""
    PERSON = "person"
    COMPANY = "company"
    PRODUCT = "product"
    LOCATION = "location"
    CONCEPT = "concept"
    ORGANIZATION = "organization"
    EVENT = "event"


class RelationshipType(str, Enum):
    """Types of relationships between entities"""
    MENTIONS = "MENTIONS"
    LINKS_TO = "LINKS_TO"
    WORKS_AT = "WORKS_AT"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    PRODUCES = "PRODUCES"


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================
# SCRAPED DATA MODELS
# ============================================

class ScrapedPage(Base):
    """Main table for scraped web pages"""
    __tablename__ = "scraped_pages"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # URL and identification
    url = Column(Text, nullable=False)
    url_hash = Column(String(64), unique=True, nullable=False, index=True)

    # Content
    title = Column(Text)
    content = Column(Text)
    content_hash = Column(String(64), unique=True, index=True)
    html_raw = Column(Text)

    # Vector embedding for semantic search
    embedding = Column(Vector(384))

    # Metadata
    metadata = Column(JSON, default={})
    tags = Column(ARRAY(String))

    # Scraping context
    scraper_type = Column(SQLEnum(ScraperType))
    proxy_used = Column(String(255))
    user_agent = Column(Text)

    # Status tracking
    status = Column(SQLEnum(PageStatus), default=PageStatus.PENDING, index=True)
    retry_count = Column(Integer, default=0)
    last_error = Column(Text)

    # Timestamps
    scraped_at = Column(TIMESTAMP, default=datetime.utcnow)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    chunks = relationship("PageChunk", back_populates="page", cascade="all, delete-orphan")
    entity_links = relationship("PageEntityLink", back_populates="page", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_pages_scraped_at', scraped_at.desc()),
        Index('idx_pages_embedding_vector', embedding, postgresql_using='hnsw',
              postgresql_with={'m': 16, 'ef_construction': 64}),
    )


class PageChunk(Base):
    """Chunks of scraped pages for better semantic search"""
    __tablename__ = "page_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    page_id = Column(Integer, ForeignKey("scraped_pages.id", ondelete="CASCADE"), nullable=False)

    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))

    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    page = relationship("ScrapedPage", back_populates="chunks")

    # Indexes
    __table_args__ = (
        Index('idx_chunks_page_id', page_id),
        Index('idx_chunks_embedding_vector', embedding, postgresql_using='hnsw',
              postgresql_with={'m': 16, 'ef_construction': 64}),
    )


# ============================================
# KNOWLEDGE GRAPH MODELS
# ============================================

class Entity(Base):
    """Entities extracted from scraped content"""
    __tablename__ = "entities"

    id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    entity_type = Column(SQLEnum(EntityType), nullable=False, index=True)
    description = Column(Text)
    metadata = Column(JSON, default={})

    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    observations = relationship("EntityObservation", back_populates="entity", cascade="all, delete-orphan")
    source_relationships = relationship(
        "Relationship",
        foreign_keys="Relationship.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan"
    )
    target_relationships = relationship(
        "Relationship",
        foreign_keys="Relationship.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan"
    )
    page_links = relationship("PageEntityLink", back_populates="entity", cascade="all, delete-orphan")


class EntityObservation(Base):
    """Observations about entities from different sources"""
    __tablename__ = "entity_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(255), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)

    observation = Column(Text, nullable=False)
    source_page_id = Column(Integer, ForeignKey("scraped_pages.id", ondelete="SET NULL"))
    confidence = Column(Float, default=1.0)

    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    entity = relationship("Entity", back_populates="observations")

    # Indexes
    __table_args__ = (
        Index('idx_observations_entity_id', entity_id),
    )


class Relationship(Base):
    """Relationships between entities"""
    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_entity_id = Column(String(255), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    target_entity_id = Column(String(255), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(SQLEnum(RelationshipType), nullable=False)

    confidence = Column(Float, default=1.0)
    metadata = Column(JSON, default={})

    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    source_entity = relationship("Entity", foreign_keys=[source_entity_id], back_populates="source_relationships")
    target_entity = relationship("Entity", foreign_keys=[target_entity_id], back_populates="target_relationships")

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint('source_entity_id', 'target_entity_id', 'relationship_type',
                        name='uq_relationship'),
        Index('idx_relationships_source', source_entity_id),
        Index('idx_relationships_target', target_entity_id),
    )


class PageEntityLink(Base):
    """Links between pages and entities"""
    __tablename__ = "page_entity_links"

    page_id = Column(Integer, ForeignKey("scraped_pages.id", ondelete="CASCADE"), primary_key=True)
    entity_id = Column(String(255), ForeignKey("entities.id", ondelete="CASCADE"), primary_key=True)

    link_type = Column(String(50), default="MENTIONS")
    confidence = Column(Float, default=1.0)

    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    page = relationship("ScrapedPage", back_populates="entity_links")
    entity = relationship("Entity", back_populates="page_links")


# ============================================
# JOB ORCHESTRATION MODELS
# ============================================

class ScrapingJob(Base):
    """Scraping jobs queue"""
    __tablename__ = "scraping_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_type = Column(String(50), nullable=False)
    start_url = Column(Text, nullable=False)
    config = Column(JSON, default={})

    # Scheduling
    scheduled_at = Column(TIMESTAMP)
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)

    # Status
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, index=True)
    pages_scraped = Column(Integer, default=0)
    pages_failed = Column(Integer, default=0)

    # Error handling
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    last_error = Column(Text)

    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationships
    url_queue_items = relationship("URLQueue", back_populates="job", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_jobs_scheduled', scheduled_at),
    )


class URLQueue(Base):
    """URL queue for distributed scraping"""
    __tablename__ = "url_queue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("scraping_jobs.id", ondelete="CASCADE"), nullable=False)

    url = Column(Text, nullable=False)
    url_hash = Column(String(64), unique=True, nullable=False, index=True)
    priority = Column(Integer, default=0, index=True)
    depth = Column(Integer, default=0)
    parent_url = Column(Text)

    # Status
    status = Column(SQLEnum(PageStatus), default=PageStatus.PENDING, index=True)
    assigned_worker = Column(String(255))
    assigned_at = Column(TIMESTAMP)

    # Retry logic
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    next_retry_at = Column(TIMESTAMP)

    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relationships
    job = relationship("ScrapingJob", back_populates="url_queue_items")

    # Indexes
    __table_args__ = (
        Index('idx_queue_status_priority', status, priority.desc()),
        Index('idx_queue_job_id', job_id),
        Index('idx_queue_next_retry', next_retry_at),
    )


# ============================================
# MONITORING & ANALYTICS MODELS
# ============================================

class Log(Base):
    """Operational logs"""
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(SQLEnum(LogLevel), nullable=False, index=True)
    message = Column(Text, nullable=False)
    component = Column(String(100))
    job_id = Column(Integer, ForeignKey("scraping_jobs.id", ondelete="SET NULL"))
    context = Column(JSON, default={})

    created_at = Column(TIMESTAMP, server_default=func.now(), index=True)

    # Indexes
    __table_args__ = (
        Index('idx_logs_created', created_at.desc()),
    )


class ScraperStats(Base):
    """Scraper performance statistics"""
    __tablename__ = "scraper_stats"

    scraper_type = Column(String(50), primary_key=True)

    total_pages = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    avg_duration_ms = Column(Float, default=0.0)

    last_used = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class CacheStats(Base):
    """Cache statistics"""
    __tablename__ = "cache_stats"

    cache_key = Column(String(255), primary_key=True)

    hit_count = Column(Integer, default=0)
    miss_count = Column(Integer, default=0)
    last_hit = Column(TIMESTAMP)

    created_at = Column(TIMESTAMP, server_default=func.now())
