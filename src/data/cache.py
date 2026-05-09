from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from config import get_settings

_DB_PATH: Path | None = None
_CONN: sqlite3.Connection | None = None


def _conn() -> sqlite3.Connection:
    global _DB_PATH, _CONN
    if _CONN is None:
        _DB_PATH = get_settings().cache_dir / "cache.sqlite"
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _CONN.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            "k TEXT PRIMARY KEY, v BLOB, ts REAL)"
        )
        _CONN.commit()
    return _CONN


def _hash_key(name: str, args: tuple, kwargs: dict) -> str:
    blob = json.dumps(
        {"name": name, "args": args, "kwargs": kwargs},
        default=str,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def disk_cache(ttl_seconds: int | None = None, namespace: str = "") -> Callable:
    """SQLite-backed function cache. ttl=None 表示永久缓存。"""

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _hash_key(f"{namespace}:{fn.__name__}", args, kwargs)
            cur = _conn().execute("SELECT v, ts FROM kv WHERE k=?", (key,))
            row = cur.fetchone()
            if row is not None:
                v, ts = row
                if ttl_seconds is None or (time.time() - ts) < ttl_seconds:
                    return pickle.loads(v)
            result = fn(*args, **kwargs)
            _conn().execute(
                "INSERT OR REPLACE INTO kv VALUES (?, ?, ?)",
                (key, pickle.dumps(result), time.time()),
            )
            _conn().commit()
            return result

        return wrapper

    return deco
