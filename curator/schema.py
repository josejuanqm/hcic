"""
Curator — Schema and Weight Engine
Implements the three primitives: Observe, Weight, Surface

Storage: SQLite + sqlite-vec
- Conceptions are the unit of memory
- Each conception has two float properties: recency and confidence
- Recency decays lazily at read time, rate proportional to confidence
- Competing conceptions coexist — contradiction creates, not overwrites
- Signal quality gates Surface, not action severity
"""

import sqlite3
import sqlite_vec
import json
import math
import time
from dataclasses import dataclass
from typing import Optional


# --- Constants ---

INITIAL_RECENCY = 1.0
INITIAL_CONFIDENCE = 0.1        # starts low, grows through signal
EXPLICIT_INSTRUCTION_MAGNITUDE = 1.0  # user correction = max contradiction
SEMANTIC_MATCH_THRESHOLD = 0.82 # cosine similarity to consider same conception
SURFACE_RECENCY_THRESHOLD = 0.15
SURFACE_CONFIDENCE_THRESHOLD = 0.05
EMBEDDING_DIM = 1024            # claude embeddings dimension


# --- Data structures ---

@dataclass
class Conception:
    id: int
    content: str
    recency: float
    confidence: float
    last_updated: float         # unix timestamp for lazy decay
    source: str                 # what observation created this


@dataclass
class SignalQuality:
    """
    Instantaneous evaluation of an incoming input.
    Not accumulated — evaluated fresh, then discarded.
    Gates Surface decisions.
    """
    score: float                # 0.0 to 1.0
    reason: str


# --- Database setup ---

def connect(db_path: str = "curator.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection):
    conn.executescript("""
        -- Core conception table
        CREATE TABLE IF NOT EXISTS conceptions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            content         TEXT NOT NULL,
            recency         REAL NOT NULL DEFAULT 1.0,
            confidence      REAL NOT NULL DEFAULT 0.1,
            last_updated    REAL NOT NULL,   -- unix timestamp
            source          TEXT,
            created_at      REAL NOT NULL
        );

        -- Vector index for semantic matching
        -- Determines when new observations relate to existing conceptions
        CREATE VIRTUAL TABLE IF NOT EXISTS conception_embeddings
        USING vec0(
            conception_id INTEGER PRIMARY KEY,
            embedding FLOAT[1024]
        );

        -- Observation log — raw inputs before they become conceptions
        CREATE TABLE IF NOT EXISTS observations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            content         TEXT NOT NULL,
            signal_quality  REAL NOT NULL,
            observed_at     REAL NOT NULL,
            resulted_in     TEXT          -- JSON list of affected conception ids
        );
    """)
    conn.commit()


# --- Weight mechanics ---

def _compute_current_recency(base_recency: float, confidence: float, last_updated: float) -> float:
    """
    Lazy decay: recency decays at a rate proportional to confidence.
    High confidence = slow decay (persistent).
    Low confidence = fast decay (momentary).

    recency(t) = base * e^(-rate * elapsed)
    rate = (1 - confidence) * decay_factor
    """
    elapsed_hours = (time.time() - last_updated) / 3600
    decay_factor = 0.05  # tune this — how fast does a zero-confidence conception fade?
    rate = (1.0 - confidence) * decay_factor
    return base_recency * math.exp(-rate * elapsed_hours)


def get_conception(conn: sqlite3.Connection, conception_id: int) -> Optional[Conception]:
    row = conn.execute(
        "SELECT * FROM conceptions WHERE id = ?", (conception_id,)
    ).fetchone()
    if not row:
        return None
    current_recency = _compute_current_recency(
        row["recency"], row["confidence"], row["last_updated"]
    )
    return Conception(
        id=row["id"],
        content=row["content"],
        recency=current_recency,
        confidence=row["confidence"],
        last_updated=row["last_updated"],
        source=row["source"]
    )


def update_weight(
    conn: sqlite3.Connection,
    conception_id: int,
    confidence_delta: float,    # positive = confirming signal, negative = contradicting
    reset_recency: bool = True  # new signal always refreshes recency
):
    """
    Signed update rule.
    Positive delta: confirming signal — confidence grows, recency resets high.
    Negative delta: contradicting signal — confidence shrinks.
    Explicit instruction: magnitude 1.0 contradiction collapses confidence immediately.
    """
    row = conn.execute(
        "SELECT recency, confidence, last_updated FROM conceptions WHERE id = ?",
        (conception_id,)
    ).fetchone()
    if not row:
        return

    current_confidence = row["confidence"]
    new_confidence = max(0.0, min(1.0, current_confidence + confidence_delta))
    new_recency = INITIAL_RECENCY if reset_recency else _compute_current_recency(
        row["recency"], current_confidence, row["last_updated"]
    )

    conn.execute(
        "UPDATE conceptions SET confidence = ?, recency = ?, last_updated = ? WHERE id = ?",
        (new_confidence, new_recency, time.time(), conception_id)
    )
    conn.commit()


def create_conception(
    conn: sqlite3.Connection,
    content: str,
    embedding: list[float],
    source: str,
    initial_confidence: float = INITIAL_CONFIDENCE
) -> int:
    """
    Create a new conception with low initial confidence, high recency.
    Contradictions don't delete — they create competing conceptions.
    Both exist simultaneously at different temporal scales.
    """
    now = time.time()
    cursor = conn.execute(
        """INSERT INTO conceptions (content, recency, confidence, last_updated, source, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (content, INITIAL_RECENCY, initial_confidence, now, source, now)
    )
    conception_id = cursor.lastrowid

    conn.execute(
        "INSERT INTO conception_embeddings (conception_id, embedding) VALUES (?, ?)",
        (conception_id, json.dumps(embedding))
    )
    conn.commit()
    return conception_id


def find_related_conceptions(
    conn: sqlite3.Connection,
    embedding: list[float],
    threshold: float = SEMANTIC_MATCH_THRESHOLD,
    limit: int = 5
) -> list[tuple[int, float]]:
    """
    Find existing conceptions semantically related to a new observation.
    Returns (conception_id, similarity) pairs above threshold.
    This is what determines: update existing confidence vs create new competing conception.
    """
    results = conn.execute(
        """SELECT conception_id, distance
           FROM conception_embeddings
           WHERE embedding MATCH ?
           AND k = ?
           ORDER BY distance""",
        (json.dumps(embedding), limit)
    ).fetchall()

    # sqlite-vec returns L2 distance, convert to similarity
    related = []
    for row in results:
        similarity = 1.0 / (1.0 + row["distance"])
        if similarity >= threshold:
            related.append((row["conception_id"], similarity))
    return related


# --- Surface ---

def surface(
    conn: sqlite3.Connection,
    signal_quality: SignalQuality,
    limit: int = 10
) -> list[Conception]:
    """
    Activates relevant context at the moment needed.
    Gated by signal quality — low quality surfaces the ambiguity itself.

    Returns conceptions above threshold, ordered by recency first
    (recency governs the present), then confidence (governs the persistent).
    """
    if signal_quality.score < 0.3:
        # Signal too ambiguous — don't resolve, surface the ambiguity
        return []

    rows = conn.execute(
        """SELECT id, content, recency, confidence, last_updated, source
           FROM conceptions
           WHERE recency >= ? AND confidence >= ?
           ORDER BY recency DESC, confidence DESC
           LIMIT ?""",
        (SURFACE_RECENCY_THRESHOLD, SURFACE_CONFIDENCE_THRESHOLD, limit)
    ).fetchall()

    # Apply lazy decay at read time, then re-sort by computed recency
    # (SQL sorted by stored recency; computed recency may differ in order)
    conceptions = []
    for row in rows:
        current_recency = _compute_current_recency(
            row["recency"], row["confidence"], row["last_updated"]
        )
        if current_recency >= SURFACE_RECENCY_THRESHOLD:
            conceptions.append(Conception(
                id=row["id"],
                content=row["content"],
                recency=current_recency,
                confidence=row["confidence"],
                last_updated=row["last_updated"],
                source=row["source"]
            ))

    conceptions.sort(key=lambda c: (c.recency, c.confidence), reverse=True)
    return conceptions


def log_observation(
    conn: sqlite3.Connection,
    content: str,
    signal_quality: float,
    affected_conceptions: list[int]
):
    conn.execute(
        """INSERT INTO observations (content, signal_quality, observed_at, resulted_in)
           VALUES (?, ?, ?, ?)""",
        (content, signal_quality, time.time(), json.dumps(affected_conceptions))
    )
    conn.commit()


if __name__ == "__main__":
    conn = connect(":memory:")
    print("Schema initialized successfully")
    print("Tables:", conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
