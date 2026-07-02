from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import jwt


APP_DIR = Path(__file__).parent
DB_PATH = Path(os.getenv("APP_DB_PATH", APP_DIR / "app_data" / "compliance_assistant.db"))
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-development-secret")
JWT_ALGORITHM = "HS256"
MIN_HS256_KEY_BYTES = 32
DB_TIMEOUT_SECONDS = 30
DB_BUSY_TIMEOUT_MS = 30_000
DB_RETRY_ATTEMPTS = 5
DB_RETRY_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class User:
    id: int
    email: str
    role: str
    created_at: str
    full_name: str = ""


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                auth_provider TEXT NOT NULL DEFAULT 'password',
                provider_subject TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS document_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                detections INTEGER NOT NULL,
                high_severity_detections INTEGER NOT NULL,
                categories_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event_type TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS detection_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                analysis_id TEXT NOT NULL,
                category TEXT NOT NULL,
                masked_value TEXT NOT NULL,
                verdict TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS ai_call_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                analysis_id TEXT NOT NULL,
                feature TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                output_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )
        _ensure_column(conn, "users", "full_name", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "users", "auth_provider", "TEXT NOT NULL DEFAULT 'password'")
        _ensure_column(conn, "users", "provider_subject", "TEXT")


def create_user(email: str, password: str, full_name: str = "") -> User:
    normalized_email = _normalize_email(email)
    _validate_password(password)
    clean_name = full_name.strip()
    salt = os.urandom(16)
    password_hash = _hash_password(password, salt)
    role = _new_user_role(normalized_email)
    created_at = _now()

    with _connect() as conn:
        try:
            cursor = conn.execute(
                """
                INSERT INTO users (full_name, email, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (clean_name, normalized_email, password_hash, _b64(salt), role, created_at),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("An account already exists for this email.") from exc
        user = User(id=int(cursor.lastrowid), email=normalized_email, role=role, created_at=created_at, full_name=clean_name)
    log_audit(user.id, "user_registered", {"email": normalized_email, "role": role, "full_name": clean_name})
    return user


def authenticate_user(email: str, password: str) -> User | None:
    normalized_email = _normalize_email(email)
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (normalized_email,)).fetchone()
        if not row:
            return None
        salt = base64.b64decode(row["salt"])
        candidate_hash = _hash_password(password, salt)
        if not hmac.compare_digest(candidate_hash, row["password_hash"]):
            return None
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (_now(), row["id"]))

    user = User(id=row["id"], email=row["email"], role=row["role"], created_at=row["created_at"], full_name=row["full_name"])
    log_audit(user.id, "login_success", {"email": user.email})
    return user


def create_or_update_oauth_user(email: str, full_name: str, provider: str, provider_subject: str) -> User:
    return _with_db_retry(
        lambda: _create_or_update_oauth_user_once(email, full_name, provider, provider_subject)
    )


def _create_or_update_oauth_user_once(email: str, full_name: str, provider: str, provider_subject: str) -> User:
    normalized_email = _normalize_email(email)
    clean_name = full_name.strip()
    clean_provider = provider.strip().lower()
    clean_subject = provider_subject.strip()
    if not clean_provider or not clean_subject:
        raise ValueError("OAuth provider details are missing.")

    audit_event = "oauth_user_registered"
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (normalized_email,)).fetchone()
        if row:
            conn.execute(
                """
                UPDATE users
                SET full_name = COALESCE(NULLIF(?, ''), full_name),
                    auth_provider = ?,
                    provider_subject = ?,
                    last_login_at = ?
                WHERE id = ?
                """,
                (clean_name, clean_provider, clean_subject, _now(), row["id"]),
            )
            updated = conn.execute("SELECT * FROM users WHERE id = ?", (row["id"],)).fetchone()
            user = _user_from_row(updated)
            audit_event = "oauth_login_success"
        else:
            salt = os.urandom(16)
            password_hash = _hash_password(secrets.token_urlsafe(32), salt)
            role = _new_user_role(normalized_email, conn)
            created_at = _now()
            cursor = conn.execute(
                """
                INSERT INTO users (
                    full_name, email, password_hash, salt, auth_provider,
                    provider_subject, role, created_at, last_login_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_name,
                    normalized_email,
                    password_hash,
                    _b64(salt),
                    clean_provider,
                    clean_subject,
                    role,
                    created_at,
                    created_at,
                ),
            )
            user = User(id=int(cursor.lastrowid), email=normalized_email, role=role, created_at=created_at, full_name=clean_name)

    log_audit(user.id, audit_event, {"email": normalized_email, "role": user.role, "provider": clean_provider})
    return user


def request_password_reset(email: str) -> str | None:
    normalized_email = _normalize_email(email)
    with _connect() as conn:
        row = conn.execute("SELECT id, email FROM users WHERE email = ?", (normalized_email,)).fetchone()
        if not row:
            return None
        token = secrets.token_urlsafe(32)
        token_hash = _token_hash(token)
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(minutes=30)).isoformat(timespec="seconds")
        conn.execute(
            """
            INSERT INTO password_reset_tokens (user_id, token_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (row["id"], token_hash, now.isoformat(timespec="seconds"), expires_at),
        )
    log_audit(row["id"], "password_reset_requested", {"email": normalized_email})
    return token


def reset_password(reset_token: str, new_password: str) -> bool:
    _validate_password(new_password)
    token_hash = _token_hash(reset_token.strip())
    now = datetime.now(timezone.utc)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, expires_at, used_at
            FROM password_reset_tokens
            WHERE token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if not row or row["used_at"]:
            return False
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at < now:
            return False
        salt = os.urandom(16)
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
            (_hash_password(new_password, salt), _b64(salt), row["user_id"]),
        )
        conn.execute(
            "UPDATE password_reset_tokens SET used_at = ? WHERE id = ?",
            (now.isoformat(timespec="seconds"), row["id"]),
        )
    log_audit(row["user_id"], "password_reset_completed", {"user_id": row["user_id"]})
    return True


def change_password(user_id: int, current_password: str, new_password: str) -> bool:
    _validate_password(new_password)
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return False
        salt = base64.b64decode(row["salt"])
        current_hash = _hash_password(current_password, salt)
        if not hmac.compare_digest(current_hash, row["password_hash"]):
            return False
        new_salt = os.urandom(16)
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
            (_hash_password(new_password, new_salt), _b64(new_salt), user_id),
        )
    log_audit(user_id, "password_changed", {"user_id": user_id})
    return True


def create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=8)).timestamp()),
    }
    return jwt.encode(payload, _jwt_signing_key(), algorithm=JWT_ALGORITHM)


def get_user_from_token(token: str) -> User | None:
    try:
        payload = jwt.decode(token, _jwt_signing_key(), algorithms=[JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except Exception:
        return None
    return get_user_by_id(user_id)


def get_user_by_id(user_id: int) -> User | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, full_name, email, role, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return _user_from_row(row)


def save_document_history(
    user_id: int,
    file_name: str,
    file_type: str,
    risk_level: str,
    risk_score: int,
    detections: int,
    high_severity_detections: int,
    categories: dict[str, int],
    metadata: dict[str, Any],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO document_history (
                user_id, file_name, file_type, risk_level, risk_score, detections,
                high_severity_detections, categories_json, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                file_name,
                file_type,
                risk_level,
                risk_score,
                detections,
                high_severity_detections,
                json.dumps(categories),
                json.dumps(metadata),
                _now(),
            ),
        )


def list_user_documents(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT file_name, file_type, risk_level, risk_score, detections,
                   high_severity_detections, categories_json, metadata_json, created_at
            FROM document_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [_document_row_to_dict(row) for row in rows]


def log_audit(user_id: int | None, event_type: str, details: dict[str, Any]) -> None:
    def write_event() -> None:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (user_id, event_type, details_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, event_type, json.dumps(details), _now()),
            )

    _with_db_retry(write_event)


def save_detection_feedback(
    user_id: int,
    analysis_id: str,
    category: str,
    masked_value: str,
    verdict: str,
    note: str = "",
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO detection_feedback (
                user_id, analysis_id, category, masked_value, verdict, note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, analysis_id, category, masked_value, verdict, note.strip(), _now()),
        )
    log_audit(user_id, "feedback_recorded", {"analysis_id": analysis_id, "category": category, "verdict": verdict})


def list_detection_feedback(user_id: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
    with _connect() as conn:
        if user_id is None:
            rows = conn.execute(
                """
                SELECT id, user_id, analysis_id, category, masked_value, verdict, note, created_at
                FROM detection_feedback
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, user_id, analysis_id, category, masked_value, verdict, note, created_at
                FROM detection_feedback
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
    return [dict(row) for row in rows]


def save_ai_call_trace(
    user_id: int | None,
    analysis_id: str,
    feature: str,
    model_provider: str,
    model_name: str,
    prompt_version: str,
    status: str,
    latency_ms: int,
    output_hash: str,
    metadata: dict[str, Any],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO ai_call_traces (
                user_id, analysis_id, feature, model_provider, model_name, prompt_version,
                status, latency_ms, output_hash, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                analysis_id,
                feature,
                model_provider,
                model_name,
                prompt_version,
                status,
                latency_ms,
                output_hash,
                json.dumps(metadata),
                _now(),
            ),
        )


def list_ai_call_traces(user_id: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
    with _connect() as conn:
        if user_id is None:
            rows = conn.execute(
                """
                SELECT id, user_id, analysis_id, feature, model_provider, model_name, prompt_version,
                       status, latency_ms, output_hash, metadata_json, created_at
                FROM ai_call_traces
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, user_id, analysis_id, feature, model_provider, model_name, prompt_version,
                       status, latency_ms, output_hash, metadata_json, created_at
                FROM ai_call_traces
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
    return [
        {
            **{key: row[key] for key in row.keys() if key != "metadata_json"},
            "metadata": json.loads(row["metadata_json"]),
        }
        for row in rows
    ]


def list_user_audits(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT event_type, details_json, created_at
            FROM audit_events
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [_audit_row_to_dict(row) for row in rows]


def admin_dashboard_data() -> dict[str, Any]:
    with _connect() as conn:
        totals = {
            "users": conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
            "documents": conn.execute("SELECT COUNT(*) FROM document_history").fetchone()[0],
            "audits": conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0],
            "high_risk_documents": conn.execute(
                "SELECT COUNT(*) FROM document_history WHERE risk_level = 'High Risk'"
            ).fetchone()[0],
        }
        users = conn.execute(
            """
            SELECT u.id, u.full_name, u.email, u.role, u.created_at, u.last_login_at,
                   COUNT(d.id) AS document_count
            FROM users u
            LEFT JOIN document_history d ON d.user_id = u.id
            GROUP BY u.id
            ORDER BY u.created_at DESC
            """
        ).fetchall()
        documents = conn.execute(
            """
            SELECT d.file_name, d.file_type, d.risk_level, d.risk_score, d.detections,
                   d.high_severity_detections, d.categories_json, d.created_at, u.email
            FROM document_history d
            JOIN users u ON u.id = d.user_id
            ORDER BY d.created_at DESC
            LIMIT 100
            """
        ).fetchall()
        audits = conn.execute(
            """
            SELECT a.event_type, a.details_json, a.created_at, u.email
            FROM audit_events a
            LEFT JOIN users u ON u.id = a.user_id
            ORDER BY a.created_at DESC
            LIMIT 100
            """
        ).fetchall()
    return {
        "totals": totals,
        "users": [dict(row) for row in users],
        "documents": [_admin_document_row_to_dict(row) for row in documents],
        "audits": [_admin_audit_row_to_dict(row) for row in audits],
    }


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=DB_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {DB_BUSY_TIMEOUT_MS}")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _with_db_retry(operation):
    last_error = None
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
            last_error = exc
            time.sleep(DB_RETRY_DELAY_SECONDS * (attempt + 1))
    raise last_error


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _jwt_signing_key() -> str | bytes:
    secret = os.getenv("JWT_SECRET", JWT_SECRET)
    key_bytes = secret.encode("utf-8")
    if len(key_bytes) >= MIN_HS256_KEY_BYTES:
        return secret
    return hashlib.sha256(key_bytes).digest()


def _user_from_row(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        email=row["email"],
        role=row["role"],
        created_at=row["created_at"],
        full_name=row["full_name"],
    )


def _new_user_role(email: str, conn: sqlite3.Connection | None = None) -> str:
    admin_email = os.getenv("ADMIN_EMAIL", "").strip().lower()
    if admin_email and email == admin_email:
        return "admin"
    if conn is not None:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        return "admin" if count == 0 else "user"
    with _connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    return "admin" if count == 0 else "user"


def _normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if "@" not in normalized or "." not in normalized.split("@")[-1]:
        raise ValueError("Enter a valid email address.")
    return normalized


def _validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return _b64(digest)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _document_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["categories"] = json.loads(result.pop("categories_json"))
    result["metadata"] = json.loads(result.pop("metadata_json"))
    return result


def _audit_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["details"] = json.loads(result.pop("details_json"))
    return result


def _admin_document_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["categories"] = json.loads(result.pop("categories_json"))
    return result


def _admin_audit_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["details"] = json.loads(result.pop("details_json"))
    return result
