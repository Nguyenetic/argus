-- Initialize PostgreSQL database with pgvector extension
-- This script runs automatically when the database container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database if not exists (already created by POSTGRES_DB env var)
-- Additional initialization can go here

-- Create indexes after table creation (handled by SQLAlchemy)
-- This file is mainly for enabling extensions

SELECT 'Database initialized successfully' AS status;
