#!/usr/bin/env python
"""Stage 3 schema migration script for SELF-OS.

Safely applies all schema changes required by Stage 3 (Agentic Functions):

    mood_snapshots:
        - stressor_tags TEXT DEFAULT '[]'
        - active_parts_keys TEXT DEFAULT '[]'
        - intervention_applied TEXT
        - feedback_score INTEGER

    journal_entries:
        - session_id TEXT
        - cognitive_load REAL

    New tables (idempotent CREATE IF NOT EXISTS):
        - psyche_states     (PsycheStateStore)
        - goals             (GoalStore)
        - agent_tasks       (TaskStore)

All ALTER TABLE statements are wrapped with proper error handling so the
script can be safely re-run even if the columns already exist.

Usage::

    python scripts/migrate_stage3.py [--db PATH]

The default database path is ``data/self_os.db``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sqlite3
import sys
from pathlib import Path

import aiosqlite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Migration definitions
# ---------------------------------------------------------------------------

_ALTER_MIGRATIONS: list[tuple[str, str]] = [
    # (table.column, SQL statement)
    (
        "mood_snapshots.stressor_tags",
        "ALTER TABLE mood_snapshots ADD COLUMN stressor_tags TEXT DEFAULT '[]'",
    ),
    (
        "mood_snapshots.active_parts_keys",
        "ALTER TABLE mood_snapshots ADD COLUMN active_parts_keys TEXT DEFAULT '[]'",
    ),
    (
        "mood_snapshots.intervention_applied",
        "ALTER TABLE mood_snapshots ADD COLUMN intervention_applied TEXT",
    ),
    (
        "mood_snapshots.feedback_score",
        "ALTER TABLE mood_snapshots ADD COLUMN feedback_score INTEGER",
    ),
    (
        "journal_entries.session_id",
        "ALTER TABLE journal_entries ADD COLUMN session_id TEXT",
    ),
    (
        "journal_entries.cognitive_load",
        "ALTER TABLE journal_entries ADD COLUMN cognitive_load REAL",
    ),
    (
        "nodes.is_deleted",
        "ALTER TABLE nodes ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0",
    ),
]

_CREATE_STATEMENTS: list[tuple[str, str]] = [
    (
        "psyche_states",
        """
        CREATE TABLE IF NOT EXISTS psyche_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            valence REAL NOT NULL DEFAULT 0.0,
            arousal REAL NOT NULL DEFAULT 0.0,
            dominance REAL NOT NULL DEFAULT 0.0,
            active_parts TEXT NOT NULL DEFAULT '[]',
            dominant_part TEXT,
            dominant_need TEXT,
            active_beliefs TEXT NOT NULL DEFAULT '[]',
            cognitive_load REAL NOT NULL DEFAULT 0.0,
            cognitive_distortions TEXT NOT NULL DEFAULT '[]',
            stressor_tags TEXT NOT NULL DEFAULT '[]',
            active_goals TEXT NOT NULL DEFAULT '[]',
            body_state TEXT,
            confidence REAL NOT NULL DEFAULT 1.0
        )
        """,
    ),
    (
        "goals",
        """
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            priority INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'active',
            parent_goal_id TEXT,
            tags TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            target_date TEXT,
            progress REAL NOT NULL DEFAULT 0.0,
            linked_node_ids TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """,
    ),
    (
        "agent_tasks",
        """
        CREATE TABLE IF NOT EXISTS agent_tasks (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            priority INTEGER NOT NULL DEFAULT 3,
            status TEXT NOT NULL DEFAULT 'pending',
            due_date TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """,
    ),
]

_INDEX_STATEMENTS: list[tuple[str, str]] = [
    (
        "idx_psyche_states_user_ts",
        """
        CREATE INDEX IF NOT EXISTS idx_psyche_states_user_ts
            ON psyche_states(user_id, timestamp DESC)
        """,
    ),
    (
        "idx_goals_user_status",
        """
        CREATE INDEX IF NOT EXISTS idx_goals_user_status
            ON goals(user_id, status)
        """,
    ),
    (
        "idx_agent_tasks_user_status",
        """
        CREATE INDEX IF NOT EXISTS idx_agent_tasks_user_status
            ON agent_tasks(user_id, status)
        """,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_migrations(db_path: str) -> None:
    """Apply all Stage 3 migrations to the SQLite database at *db_path*."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to database: %s", path)
    async with aiosqlite.connect(str(path)) as conn:
        # 1. CREATE TABLE IF NOT EXISTS
        for table_name, sql in _CREATE_STATEMENTS:
            logger.info("Creating table (if not exists): %s", table_name)
            await conn.execute(sql)

        # 2. CREATE INDEX IF NOT EXISTS
        for index_name, sql in _INDEX_STATEMENTS:
            logger.info("Creating index (if not exists): %s", index_name)
            await conn.execute(sql)

        # 3. ALTER TABLE (suppress if column already exists)
        for label, sql in _ALTER_MIGRATIONS:
            with contextlib.suppress(sqlite3.OperationalError):
                await conn.execute(sql)
                logger.info("Applied migration: %s", label)

        await conn.commit()

    logger.info("Stage 3 migration complete.")


def main() -> None:
    """Entry point for the migration script."""
    parser = argparse.ArgumentParser(description="SELF-OS Stage 3 schema migration")
    parser.add_argument(
        "--db",
        default="data/self_os.db",
        help="Path to the SQLite database file (default: data/self_os.db)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_migrations(args.db))
    except KeyboardInterrupt:
        logger.info("Migration interrupted.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Migration failed: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
