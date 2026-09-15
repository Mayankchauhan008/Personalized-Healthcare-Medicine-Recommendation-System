from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_DIR = ROOT / "instance"
DB_PATH = INSTANCE_DIR / "medireco_users.db"


def get_connection() -> sqlite3.Connection:
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def normalize_email(email: str) -> str:
    return email.strip().lower()


def create_user(full_name: str, email: str, password: str) -> tuple[bool, str]:
    full_name = " ".join(full_name.strip().split())
    email = normalize_email(email)

    if len(full_name) < 2:
        return False, "Please enter your full name."
    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
                (full_name, email, generate_password_hash(password)),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

    return True, "Registration successful. You can now log in."


def authenticate(email: str, password: str) -> Optional[sqlite3.Row]:
    email = normalize_email(email)
    with get_connection() as conn:
        user = conn.execute(
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if user and check_password_hash(user["password_hash"], password):
        return user
    return None
