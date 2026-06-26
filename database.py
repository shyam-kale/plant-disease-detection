import re
import time
import threading
import logging
from datetime import datetime

import pymysql
import pymysql.cursors
from flask import g
from config import Config

logger = logging.getLogger("spinach")

_pool: list = []
_pool_lock = threading.Lock()


def _new_conn(db: str = None) -> pymysql.connections.Connection:
    return pymysql.connect(
        host=Config.DB_HOST,
        port=Config.DB_PORT,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        database=db or Config.DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=Config.DB_CONNECT_TIMEOUT,
        autocommit=False,
    )


def _get_conn() -> pymysql.connections.Connection:
    with _pool_lock:
        while _pool:
            conn = _pool.pop()
            try:
                conn.ping(reconnect=True)
                return conn
            except Exception:
                pass
    return _new_conn()


def _return_conn(conn: pymysql.connections.Connection) -> None:
    with _pool_lock:
        if len(_pool) < Config.DB_POOL_SIZE:
            _pool.append(conn)
        else:
            try:
                conn.close()
            except Exception:
                pass


def init_db() -> None:
    if not Config.DB_HOST or not Config.DB_PASSWORD:
        logger.error(
            "DB_HOST or DB_PASSWORD is not set in .env — "
            "AWS RDS connection cannot be established."
        )
        return

    for attempt in range(1, 6):
        try:
            # Connect without specifying a database first so we can CREATE it
            conn = _new_conn(db="")
            with conn.cursor() as cur:
                safe_name = re.sub(r"[^\w]", "_", Config.DB_NAME)
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{safe_name}` "
                    f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
                cur.execute(f"USE `{safe_name}`")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id                 INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                        image_name         VARCHAR(255)      NOT NULL,
                        prediction_result  VARCHAR(100)      NOT NULL,
                        confidence         FLOAT             NOT NULL,
                        model_used         VARCHAR(60)       NOT NULL,
                        file_hash          CHAR(64)          NOT NULL,
                        file_size          INT UNSIGNED      NOT NULL,
                        original_width     SMALLINT UNSIGNED NOT NULL,
                        original_height    SMALLINT UNSIGNED NOT NULL,
                        top3_predictions   TEXT              NULL,
                        all_probabilities  TEXT              NULL,
                        processing_time_ms FLOAT             NOT NULL,
                        thumbnail          MEDIUMTEXT        NULL,
                        image_data         LONGBLOB          NULL,
                        created_at         TIMESTAMP         NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_created  (created_at),
                        INDEX idx_result   (prediction_result),
                        INDEX idx_hash     (file_hash),
                        INDEX idx_model    (model_used)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id             INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
                        prediction_id  INT UNSIGNED NOT NULL,
                        correct_label  VARCHAR(100) NOT NULL,
                        user_comment   VARCHAR(500) NOT NULL DEFAULT '',
                        created_at     TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uq_pred (prediction_id),
                        CONSTRAINT fk_feedback_pred
                            FOREIGN KEY (prediction_id)
                            REFERENCES predictions(id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """)
            conn.commit()
            conn.close()
            logger.info(
                "AWS RDS ready — %s:%s/%s",
                Config.DB_HOST, Config.DB_PORT, Config.DB_NAME,
            )
            return
        except Exception as exc:
            logger.warning("RDS init attempt %d/5: %s", attempt, exc)
            if attempt < 5:
                time.sleep(2 ** attempt)
            else:
                logger.error(
                    "Cannot reach AWS RDS after 5 attempts. "
                    "Check DB_HOST, DB_USER, DB_PASSWORD and RDS security group inbound rules (port 3306 open)."
                )


threading.Thread(target=init_db, daemon=True).start()


def get_db() -> pymysql.connections.Connection:
    if "db" not in g:
        if not Config.DB_HOST or not Config.DB_PASSWORD:
            raise RuntimeError(
                "Database not configured. "
                "Set DB_HOST, DB_USER, DB_PASSWORD, DB_NAME in .env and restart."
            )
        try:
            g.db = _get_conn()
        except Exception as exc:
            logger.error("RDS connection failed: %s", exc)
            raise RuntimeError(
                f"Cannot connect to AWS RDS ({Config.DB_HOST}). "
                f"Check your .env credentials and RDS security group (port 3306 inbound). "
                f"Detail: {exc}"
            ) from exc
    return g.db


def close_db(exc=None) -> None:
    conn = g.pop("db", None)
    if conn is None:
        return
    if exc:
        try:
            conn.rollback()
        except Exception:
            pass
    _return_conn(conn)


def execute(sql: str, params=None, fetch=False, fetchone=False, commit=False):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            if commit:
                conn.commit()
            if fetchone:
                return cur.fetchone()
            if fetch:
                return cur.fetchall()
            return cur.lastrowid
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error("DB error [%.120s]: %s", sql, exc)
        raise


class PredictionDAO:

    @staticmethod
    def insert(image_name, prediction, confidence, model_used, file_hash,
               file_size, width, height, top3_json, all_proba_json,
               processing_ms, thumbnail=None, image_data=None) -> int:
        return execute(
            "INSERT INTO predictions "
            "(image_name, prediction_result, confidence, model_used, file_hash, "
            " file_size, original_width, original_height, top3_predictions, "
            " all_probabilities, processing_time_ms, thumbnail, image_data) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                str(image_name), str(prediction), float(confidence),
                str(model_used), str(file_hash), int(file_size),
                int(width), int(height), str(top3_json), str(all_proba_json),
                float(processing_ms), thumbnail, image_data,
            ),
            commit=True,
        )

    @staticmethod
    def get_by_id(pid: int):
        return execute(
            "SELECT p.*, f.correct_label AS feedback_label "
            "FROM predictions p "
            "LEFT JOIN feedback f ON f.prediction_id = p.id "
            "WHERE p.id = %s",
            (pid,), fetchone=True,
        )

    @staticmethod
    def get_by_hash(file_hash: str):
        return execute(
            "SELECT * FROM predictions "
            "WHERE file_hash = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (file_hash,), fetchone=True,
        )

    @staticmethod
    def delete(pid: int) -> None:
        execute("DELETE FROM predictions WHERE id = %s", (pid,), commit=True)

    @staticmethod
    def paginate(page=1, per_page=10, label=None, model=None) -> dict:
        where, params = [], []
        if label:
            where.append("p.prediction_result = %s")
            params.append(label)
        if model:
            where.append("p.model_used = %s")
            params.append(model)
        wsql = ("WHERE " + " AND ".join(where)) if where else ""

        total_row = execute(
            f"SELECT COUNT(*) AS cnt FROM predictions p {wsql}",
            params, fetchone=True,
        ) or {}
        total = total_row.get("cnt", 0)

        offset = (page - 1) * per_page
        rows = execute(
            f"SELECT p.id, p.image_name, p.prediction_result, p.confidence, "
            f"       p.model_used, p.processing_time_ms, p.created_at, "
            f"       p.top3_predictions, p.all_probabilities, "
            f"       p.file_size, p.original_width, p.original_height, "
            f"       p.file_hash, p.thumbnail, "
            f"       f.correct_label AS feedback_label "
            f"FROM predictions p "
            f"LEFT JOIN feedback f ON f.prediction_id = p.id "
            f"{wsql} "
            f"ORDER BY p.created_at DESC "
            f"LIMIT %s OFFSET %s",
            params + [per_page, offset], fetch=True,
        )
        return {
            "data":        rows or [],
            "total":       total,
            "page":        page,
            "per_page":    per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }

    @staticmethod
    def search(query: str, limit=30):
        like = f"%{query}%"
        return execute(
            "SELECT id, image_name, prediction_result, confidence, "
            "       model_used, processing_time_ms, created_at "
            "FROM predictions "
            "WHERE image_name LIKE %s OR prediction_result LIKE %s "
            "ORDER BY created_at DESC "
            "LIMIT %s",
            (like, like, limit), fetch=True,
        )

    @staticmethod
    def get_image_data(pid: int):
        return execute(
            "SELECT image_name, image_data FROM predictions WHERE id = %s",
            (pid,), fetchone=True,
        )

    @staticmethod
    def stats() -> dict:
        total = (execute(
            "SELECT COUNT(*) AS cnt FROM predictions",
            fetchone=True,
        ) or {}).get("cnt", 0)

        today = (execute(
            "SELECT COUNT(*) AS cnt FROM predictions WHERE DATE(created_at) = CURDATE()",
            fetchone=True,
        ) or {}).get("cnt", 0)

        week = (execute(
            "SELECT COUNT(*) AS cnt FROM predictions "
            "WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)",
            fetchone=True,
        ) or {}).get("cnt", 0)

        avg_row = execute(
            "SELECT AVG(processing_time_ms) AS v FROM predictions",
            fetchone=True,
        ) or {}
        avg_ms = round(float(avg_row.get("v") or 0), 1)

        by_lbl = execute(
            "SELECT prediction_result AS label, "
            "       COUNT(*) AS cnt, "
            "       AVG(confidence) AS avg_conf, "
            "       MIN(confidence) AS min_conf, "
            "       MAX(confidence) AS max_conf "
            "FROM predictions "
            "GROUP BY prediction_result "
            "ORDER BY cnt DESC",
            fetch=True,
        ) or []

        by_mdl = execute(
            "SELECT model_used, COUNT(*) AS cnt "
            "FROM predictions "
            "GROUP BY model_used",
            fetch=True,
        ) or []

        return {
            "total":    total,
            "today":    today,
            "week":     week,
            "avg_ms":   avg_ms,
            "by_label": by_lbl,
            "by_model": by_mdl,
        }

    @staticmethod
    def timeline(days=7) -> list:
        rows = execute(
            "SELECT DATE(created_at) AS day, "
            "       COUNT(*) AS cnt, "
            "       AVG(confidence) AS avg_conf "
            "FROM predictions "
            "WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY) "
            "GROUP BY DATE(created_at) "
            "ORDER BY day ASC",
            (days,), fetch=True,
        ) or []
        return [
            {
                "day":            str(r["day"]),
                "count":          r["cnt"],
                "avg_confidence": round(float(r["avg_conf"] or 0), 1),
            }
            for r in rows
        ]

    @staticmethod
    def export_rows(label=None, model=None) -> list:
        where, params = [], []
        if label:
            where.append("prediction_result = %s")
            params.append(label)
        if model:
            where.append("model_used = %s")
            params.append(model)
        wsql = ("WHERE " + " AND ".join(where)) if where else ""
        return execute(
            f"SELECT id, image_name, prediction_result, confidence, model_used, "
            f"       file_size, original_width, original_height, "
            f"       processing_time_ms, created_at "
            f"FROM predictions {wsql} "
            f"ORDER BY created_at DESC",
            params, fetch=True,
        ) or []


class FeedbackDAO:

    @staticmethod
    def insert(prediction_id: int, correct_label: str, comment: str = "") -> None:
        execute(
            "INSERT INTO feedback (prediction_id, correct_label, user_comment) "
            "VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "correct_label = VALUES(correct_label), "
            "user_comment  = VALUES(user_comment)",
            (prediction_id, correct_label, comment), commit=True,
        )

    @staticmethod
    def get_all(limit=50) -> list:
        return execute(
            "SELECT f.id, f.prediction_id, f.correct_label, f.user_comment, "
            "       f.created_at, p.image_name, p.prediction_result "
            "FROM feedback f "
            "JOIN predictions p ON f.prediction_id = p.id "
            "ORDER BY f.created_at DESC "
            "LIMIT %s",
            (limit,), fetch=True,
        ) or []

    @staticmethod
    def accuracy() -> dict:
        row = execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN f.correct_label = p.prediction_result THEN 1 ELSE 0 END) AS correct "
            "FROM feedback f "
            "JOIN predictions p ON f.prediction_id = p.id",
            fetchone=True,
        ) or {}
        total   = row.get("total") or 0
        correct = row.get("correct") or 0
        return {
            "total_feedback": total,
            "correct":        correct,
            "accuracy_pct":   round(100 * correct / total, 1) if total else 0,
        }
