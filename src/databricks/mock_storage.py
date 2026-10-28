"""Local fallback storage matching Databricks Delta Lake schema for offline use and tests."""

import sqlite3
import time
from typing import Any, Dict, List, Optional

from src.engine.game_state import MatchStats, ShotRecord


class LocalDeltaStorage:
    """SQLite-backed storage mirroring Databricks Delta Lake tables."""

    def __init__(self, db_path: str = "ping_pong_local.db"):
        self.db_path = db_path
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Create Delta-compatible relational schemas."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Matches table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ping_pong_matches (
                    match_id TEXT PRIMARY KEY,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    player_score INTEGER NOT NULL,
                    ai_score INTEGER NOT NULL,
                    winner TEXT,
                    difficulty TEXT NOT NULL,
                    total_rallies INTEGER NOT NULL,
                    longest_rally INTEGER NOT NULL,
                    avg_speed_kmh REAL NOT NULL,
                    player_smashes INTEGER NOT NULL
                )
            """)
            # Shots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ping_pong_shots (
                    shot_id TEXT PRIMARY KEY,
                    match_id TEXT NOT NULL,
                    shot_number INTEGER NOT NULL,
                    hitter TEXT NOT NULL,
                    shot_type TEXT NOT NULL,
                    speed_kmh REAL NOT NULL,
                    spin TEXT NOT NULL,
                    landing_x REAL,
                    landing_z REAL,
                    result TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (match_id) REFERENCES ping_pong_matches(match_id)
                )
            """)
            conn.commit()

    def save_match(self, stats: MatchStats):
        """Persist match metadata and its shots."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ping_pong_matches (
                    match_id, start_time, end_time, player_score, ai_score,
                    winner, difficulty, total_rallies, longest_rally,
                    avg_speed_kmh, player_smashes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats.match_id,
                stats.start_time,
                stats.end_time or time.time(),
                stats.player_score,
                stats.ai_score,
                stats.winner or "IN_PROGRESS",
                stats.difficulty,
                stats.total_rallies,
                stats.longest_rally,
                stats.avg_speed_kmh,
                stats.player_smashes,
            ))

            for s in stats.shots:
                cursor.execute("""
                    INSERT OR REPLACE INTO ping_pong_shots (
                        shot_id, match_id, shot_number, hitter, shot_type,
                        speed_kmh, spin, landing_x, landing_z, result, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    s.shot_id,
                    s.match_id,
                    s.shot_number,
                    s.hitter,
                    s.shot_type,
                    s.speed_kmh,
                    s.spin,
                    s.landing_x,
                    s.landing_z,
                    s.result,
                    s.timestamp,
                ))
            conn.commit()

    def get_matches(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent match records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ping_pong_matches
                ORDER BY start_time DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_shots_by_match(self, match_id: str) -> List[Dict[str, Any]]:
        """Retrieve all recorded shots for a given match."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ping_pong_shots
                WHERE match_id = ?
                ORDER BY shot_number ASC
            """, (match_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_aggregated_stats(self) -> Dict[str, Any]:
        """Compute lifetime telemetry aggregates."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_matches,
                    SUM(CASE WHEN winner = 'PLAYER' THEN 1 ELSE 0 END) as player_wins,
                    SUM(CASE WHEN winner IN ('AI', 'ROBERVAL') THEN 1 ELSE 0 END) as ai_wins,
                    AVG(avg_speed_kmh) as lifetime_avg_speed,
                    MAX(longest_rally) as record_rally,
                    SUM(player_smashes) as total_smashes
                FROM ping_pong_matches
            """)
            row = cursor.fetchone()
            if row and row["total_matches"] > 0:
                total = row["total_matches"]
                p_wins = row["player_wins"] or 0
                win_rate = (p_wins / total) * 100.0 if total > 0 else 0.0
                return {
                    "total_matches": total,
                    "player_wins": p_wins,
                    "ai_wins": row["ai_wins"] or 0,
                    "win_rate_pct": round(win_rate, 1),
                    "avg_speed_kmh": round(row["lifetime_avg_speed"] or 0.0, 1),
                    "record_rally": row["record_rally"] or 0,
                    "total_smashes": row["total_smashes"] or 0,
                }
            return {
                "total_matches": 0,
                "player_wins": 0,
                "ai_wins": 0,
                "win_rate_pct": 0.0,
                "avg_speed_kmh": 0.0,
                "record_rally": 0,
                "total_smashes": 0,
            }
