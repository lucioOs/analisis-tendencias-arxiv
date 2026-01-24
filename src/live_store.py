from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

DB_PATH = Path("data/live/live.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  uid TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  published TEXT,
  title TEXT,
  text TEXT,
  url TEXT,
  raw_json TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_items_source ON items(source);
CREATE INDEX IF NOT EXISTS idx_items_published ON items(published);
"""

STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_state (
  source TEXT PRIMARY KEY,
  etag TEXT,
  last_modified TEXT,
  updated_at TEXT DEFAULT (datetime('now'))
);
"""

def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH.as_posix())
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.executescript(SCHEMA)
    con.executescript(STATE_SCHEMA)
    return con

def upsert_item(con: sqlite3.Connection, item: Dict[str, Any]) -> bool:
    """
    Inserta si no existe. Regresa True si insertó (nuevo), False si ya estaba.
    """
    cur = con.cursor()
    try:
        cur.execute(
            """
            INSERT INTO items(uid, source, published, title, text, url, raw_json)
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                item.get("uid"),
                item.get("source"),
                item.get("published"),
                item.get("title"),
                item.get("text"),
                item.get("url"),
                item.get("raw_json"),
            ),
        )
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def get_state(con: sqlite3.Connection, source: str) -> dict:
    cur = con.cursor()
    cur.execute("SELECT etag, last_modified FROM fetch_state WHERE source=?", (source,))
    row = cur.fetchone()
    if not row:
        return {"etag": None, "last_modified": None}
    return {"etag": row[0], "last_modified": row[1]}

def set_state(con: sqlite3.Connection, source: str, etag: Optional[str], last_modified: Optional[str]) -> None:
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO fetch_state(source, etag, last_modified) VALUES(?,?,?)
        ON CONFLICT(source) DO UPDATE SET etag=excluded.etag, last_modified=excluded.last_modified, updated_at=datetime('now')
        """,
        (source, etag, last_modified),
    )
    con.commit()
