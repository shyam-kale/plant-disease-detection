-- ─────────────────────────────────────────────
-- IMAGE CLASSIFIER DATABASE SCHEMA
-- MySQL 8 compatible
-- ─────────────────────────────────────────────

CREATE DATABASE IF NOT EXISTS image_classifier
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE image_classifier;

-- ─────────────────────────────────────────────
-- PREDICTIONS TABLE
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    image_name          VARCHAR(255)    NOT NULL,
    prediction_result   VARCHAR(100)    NOT NULL,
    confidence          FLOAT           NOT NULL DEFAULT 0,
    model_used          VARCHAR(60)     NOT NULL DEFAULT 'random_forest',
    file_hash           CHAR(64)        NOT NULL DEFAULT '',
    file_size           INT UNSIGNED    NOT NULL DEFAULT 0,
    original_width      SMALLINT UNSIGNED NOT NULL DEFAULT 0,
    original_height     SMALLINT UNSIGNED NOT NULL DEFAULT 0,
    top3_predictions    JSON,
    all_probabilities   JSON,
    feature_vector      JSON,
    processing_time_ms  FLOAT           NOT NULL DEFAULT 0,
    feature_version     VARCHAR(20)     NOT NULL DEFAULT 'v2.0',
    thumbnail           MEDIUMTEXT,
    created_at          TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_created_at        (created_at),
    INDEX idx_prediction_result (prediction_result),
    INDEX idx_model_used        (model_used),
    INDEX idx_file_hash         (file_hash),
    INDEX idx_confidence        (confidence),
    INDEX idx_result_model      (prediction_result, model_used)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ─────────────────────────────────────────────
-- FEEDBACK TABLE
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id              INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    prediction_id   INT UNSIGNED    NOT NULL,
    correct_label   VARCHAR(100)    NOT NULL,
    user_comment    VARCHAR(500)    NOT NULL DEFAULT '',
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uq_prediction (prediction_id),
    INDEX idx_correct_label (correct_label),
    CONSTRAINT fk_feedback_prediction
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ─────────────────────────────────────────────
-- USEFUL VIEWS
-- ─────────────────────────────────────────────
CREATE OR REPLACE VIEW v_prediction_summary AS
SELECT
    p.id,
    p.image_name,
    p.prediction_result,
    ROUND(p.confidence, 2)          AS confidence_pct,
    p.model_used,
    p.file_size,
    p.original_width,
    p.original_height,
    p.processing_time_ms,
    p.thumbnail,
    p.created_at,
    f.correct_label                 AS feedback_label,
    (f.correct_label = p.prediction_result) AS was_correct
FROM predictions p
LEFT JOIN feedback f ON f.prediction_id = p.id;

CREATE OR REPLACE VIEW v_label_stats AS
SELECT
    prediction_result               AS label,
    COUNT(*)                        AS total,
    ROUND(AVG(confidence), 2)       AS avg_confidence,
    ROUND(MIN(confidence), 2)       AS min_confidence,
    ROUND(MAX(confidence), 2)       AS max_confidence,
    ROUND(AVG(processing_time_ms), 2) AS avg_processing_ms
FROM predictions
GROUP BY prediction_result
ORDER BY total DESC;

CREATE OR REPLACE VIEW v_daily_stats AS
SELECT
    DATE(created_at)                AS day,
    COUNT(*)                        AS total_predictions,
    ROUND(AVG(confidence), 2)       AS avg_confidence,
    COUNT(DISTINCT model_used)      AS models_used
FROM predictions
GROUP BY DATE(created_at)
ORDER BY day DESC;
