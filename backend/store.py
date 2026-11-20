"""Append-only metadata with atomically written server-owned artifacts."""

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

import joblib
import pandas as pd

ROOT = Path(
    os.environ.get("FAIRPRICE_DATA", Path(__file__).resolve().parents[1] / "data")
)
ROOT.mkdir(parents=True, exist_ok=True)
_SCHEMA_LOCK = Lock()
_INITIALIZED = set()


@contextmanager
def connection():
    db = sqlite3.connect(ROOT / "fairprice.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    try:
        with _SCHEMA_LOCK:
            if ROOT not in _INITIALIZED:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT, created TEXT, body TEXT)"
                )
                db.commit()
                _INITIALIZED.add(ROOT)
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def save(kind, body, *, writer=None, suffix=None):
    uid = uuid4().hex
    item = {"id": uid, "created": datetime.now(UTC).isoformat(), **body}
    artifact = ROOT / f"{uid}{suffix}" if writer else None
    temporary = ROOT / f".{uid}.tmp" if writer else None
    try:
        if writer:
            writer(temporary)
            item["artifact_sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
            temporary.replace(artifact)
        with connection() as db:
            db.execute(
                "INSERT INTO records VALUES (?,?,?,?)",
                (uid, kind, item["created"], json.dumps(item, allow_nan=False)),
            )
    except Exception:
        if temporary:
            temporary.unlink(missing_ok=True)
            artifact.unlink(missing_ok=True)
        raise
    return item


def get(uid, kind):
    with connection() as db:
        row = db.execute(
            "SELECT body FROM records WHERE id=? AND kind=?", (uid, kind)
        ).fetchone()
    if not row:
        raise KeyError(uid)
    return json.loads(row["body"])


def listing(kind):
    with connection() as db:
        return [
            json.loads(r["body"])
            for r in db.execute(
                "SELECT body FROM records WHERE kind=? ORDER BY created DESC", (kind,)
            ).fetchall()
        ]


def artifact_path(uid, kind, suffix):
    record = get(uid, kind)
    path = ROOT / f"{uid}{suffix}"
    if not path.is_file():
        raise ValueError(
            "Saved artifact is missing; restore its backup or recreate this run."
        )
    if (
        record.get("artifact_sha256")
        and hashlib.sha256(path.read_bytes()).hexdigest() != record["artifact_sha256"]
    ):
        raise ValueError(
            "Saved artifact failed its integrity check; recreate this run."
        )
    return path


def frame(uid):
    return pd.read_csv(
        artifact_path(uid, "dataset", ".csv"), dtype={"transaction_id": str}
    )


def model(uid):
    # Never accept model uploads; only deserialize our server-owned artifacts.
    return joblib.load(artifact_path(uid, "experiment", ".joblib"))
