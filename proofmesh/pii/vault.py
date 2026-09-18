"""
PII Vault Implementation using SQLite3.
Provides thread-safe, persistent local storage for raw PII values and token mappings.
"""

import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional, Union


class PIIVault:
    """
    Local SQLite Vault for storing raw PII and corresponding entity tokens.
    Guarantees thread-safe access across concurrent threads/workers.
    """

    def __init__(self, db_path: Union[str, Path] = "pii_vault.db"):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        # Ensure parent directory exists if a path with directories was passed
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS pii_vault (
                            entity_id TEXT PRIMARY KEY,
                            raw_value TEXT UNIQUE NOT NULL,
                            entity_type TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_raw_value ON pii_vault (raw_value);"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_entity_type ON pii_vault (entity_type);"
                    )
            finally:
                conn.close()

    def get_or_create_token(self, raw_value: str, entity_type: str) -> str:
        """
        Normalizes whitespace. If raw_value already has an assigned token, returns existing entity_id.
        If new, assigns next sequential token for that entity_type (e.g. [PERSON_001]), persists, and returns it.
        """
        raw_clean = " ".join(raw_value.strip().split())
        entity_type_clean = entity_type.upper().strip()

        with self._lock:
            conn = self._get_connection()
            try:
                # 1. Lookup existing token
                cursor = conn.execute(
                    "SELECT entity_id FROM pii_vault WHERE raw_value = ?",
                    (raw_clean,),
                )
                row = cursor.fetchone()
                if row:
                    return row["entity_id"]

                # 2. Sequential count for new token
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM pii_vault WHERE entity_type = ?",
                    (entity_type_clean,),
                )
                count = cursor.fetchone()["cnt"] + 1
                token_id = f"[{entity_type_clean}_{count:03d}]"

                while True:
                    cur = conn.execute(
                        "SELECT 1 FROM pii_vault WHERE entity_id = ?", (token_id,)
                    )
                    if not cur.fetchone():
                        break
                    count += 1
                    token_id = f"[{entity_type_clean}_{count:03d}]"

                # 3. Store record
                with conn:
                    conn.execute(
                        """
                        INSERT INTO pii_vault (entity_id, raw_value, entity_type)
                        VALUES (?, ?, ?)
                        """,
                        (token_id, raw_clean, entity_type_clean),
                    )
                return token_id
            finally:
                conn.close()

    def get_raw_value(self, entity_id: str) -> Optional[str]:
        """
        Reverse lookup for local de-redaction. Returns raw PII string or None if missing.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT raw_value FROM pii_vault WHERE entity_id = ?",
                    (entity_id.strip(),),
                )
                row = cursor.fetchone()
                return row["raw_value"] if row else None
            finally:
                conn.close()

    def get_all_entries(self) -> Dict[str, str]:
        """
        Returns a dict mapping raw_value -> entity_type for all vault records.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT raw_value, entity_type FROM pii_vault;")
                return {row["raw_value"]: row["entity_type"] for row in cursor.fetchall()}
            finally:
                conn.close()

    def get_all_mappings(self) -> Dict[str, str]:
        """
        Returns a dict mapping raw_value -> entity_id for backward compatibility.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT raw_value, entity_id FROM pii_vault;")
                return {row["raw_value"]: row["entity_id"] for row in cursor.fetchall()}
            finally:
                conn.close()

    def get_all_reverse_mappings(self) -> Dict[str, str]:
        """
        Returns a dict mapping entity_id -> raw_value for backward compatibility.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT entity_id, raw_value FROM pii_vault;")
                return {row["entity_id"]: row["raw_value"] for row in cursor.fetchall()}
            finally:
                conn.close()

    def clear_vault(self) -> None:
        """
        Clears all vault table records.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("DELETE FROM pii_vault;")
            finally:
                conn.close()
