-- =============================================================================
-- JARVIS — PostgreSQL initialization script
-- Executed once when the postgres container starts for the first time.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Connection settings
-- ---------------------------------------------------------------------------
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------

-- Universally unique identifiers (gen_random_uuid(), uuid_generate_v4(), etc.)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Trigram-based text search and similarity matching (used by memory search)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Cryptographic functions (pgcrypto — useful for application-level encryption)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- Timezone
-- ---------------------------------------------------------------------------
ALTER DATABASE jarvisdb SET timezone TO 'UTC';

-- ---------------------------------------------------------------------------
-- Roles
-- ---------------------------------------------------------------------------

-- readonly: used by analytics / BI tools; SELECT only, no DML
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jarvis_readonly') THEN
        CREATE ROLE jarvis_readonly NOLOGIN NOINHERIT;
    END IF;
END
$$;

-- Read-write application role (owns all objects created by migrations)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jarvis') THEN
        CREATE ROLE jarvis LOGIN PASSWORD 'jarvispassword';
    END IF;
END
$$;

-- Grant readonly role the ability to SELECT on all present and future tables
GRANT CONNECT ON DATABASE jarvisdb TO jarvis_readonly;
GRANT USAGE ON SCHEMA public TO jarvis_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO jarvis_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON SEQUENCES TO jarvis_readonly;

-- ---------------------------------------------------------------------------
-- Schema ownership
-- ---------------------------------------------------------------------------
GRANT ALL PRIVILEGES ON DATABASE jarvisdb TO jarvis;
GRANT ALL PRIVILEGES ON SCHEMA public TO jarvis;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO jarvis;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO jarvis;

-- ---------------------------------------------------------------------------
-- Performance / configuration tweaks applied to the database
-- ---------------------------------------------------------------------------

-- Use the statistics target for better planner estimates on jsonb columns
ALTER DATABASE jarvisdb SET default_statistics_target = 200;

-- Enable parallel queries (set to appropriate fraction of max_workers_per_gather)
ALTER DATABASE jarvisdb SET max_parallel_workers_per_gather = 2;

-- ---------------------------------------------------------------------------
-- Utility functions
-- ---------------------------------------------------------------------------

-- updated_at trigger function — used by Alembic-managed tables
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW() AT TIME ZONE 'UTC';
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Search vector refresh helper (used by full-text search on messages table)
CREATE OR REPLACE FUNCTION refresh_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    -- Overridden per table; placeholder function so dependent triggers compile.
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON DATABASE jarvisdb IS 'JARVIS AI Assistant — primary application database';
COMMENT ON EXTENSION "uuid-ossp"   IS 'RFC 4122 UUID generation functions';
COMMENT ON EXTENSION pg_trgm       IS 'Trigram similarity matching for fuzzy text search';
COMMENT ON EXTENSION pgcrypto      IS 'Cryptographic functions (hashing, encryption)';
COMMENT ON ROLE jarvis_readonly    IS 'Analytics / BI read-only access role';
