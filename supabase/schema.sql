-- ==============================================================================
-- Churn MLOps Platform - Supabase PostgreSQL Schema
-- ==============================================================================

-- 1. Customers Table: Stores cleaned raw customer records and labels
CREATE TABLE IF NOT EXISTS customers (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT UNIQUE NOT NULL,
    features JSONB NOT NULL,
    churn_label BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast customer lookup by customer_id
CREATE INDEX IF NOT EXISTS idx_customers_customer_id ON customers (customer_id);

-- 2. Predictions Table: Logs real-time and batch churn inference records
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT REFERENCES customers(customer_id) ON DELETE CASCADE,
    churn_probability FLOAT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for querying predictions by customer and high-risk threshold
CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON predictions (customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_churn_probability ON predictions (churn_probability);
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at ON predictions (predicted_at DESC);

-- 3. Model Runs Table: Tracks MLflow training runs, model metadata, and key metrics
CREATE TABLE IF NOT EXISTS model_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    roc_auc FLOAT NOT NULL,
    f1 FLOAT NOT NULL,
    trained_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for ordering latest model runs
CREATE INDEX IF NOT EXISTS idx_model_runs_trained_at ON model_runs (trained_at DESC);
