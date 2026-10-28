"""Unit tests for Databricks storage, Delta tables fallback, and analytics."""

import os
import time
import pytest

from src.engine.game_state import MatchStats, ShotRecord
from src.databricks.mock_storage import LocalDeltaStorage
from src.databricks.analytics import PingPongAnalytics


@pytest.fixture
def temp_storage(tmp_path):
    db_file = tmp_path / "test_ping_pong.db"
    return LocalDeltaStorage(db_path=str(db_file))


def test_save_and_retrieve_match(temp_storage):
    match_id = "test-match-123"
    stats = MatchStats(
        match_id=match_id,
        start_time=time.time(),
        end_time=time.time() + 60,
        player_score=11,
        ai_score=8,
        winner="PLAYER",
        difficulty="MEDIUM",
        total_rallies=15,
        longest_rally=7,
        avg_speed_kmh=24.5,
        player_smashes=3,
    )

    shot1 = ShotRecord(
        shot_id="s1",
        match_id=match_id,
        shot_number=1,
        hitter="PLAYER",
        shot_type="SERVE",
        speed_kmh=18.0,
        spin="NONE",
        landing_x=0.1,
        landing_z=0.8,
        result="IN_PLAY",
    )
    shot2 = ShotRecord(
        shot_id="s2",
        match_id=match_id,
        shot_number=2,
        hitter="AI",
        shot_type="DRIVE",
        speed_kmh=22.0,
        spin="NONE",
        landing_x=-0.2,
        landing_z=-0.7,
        result="IN_PLAY",
    )
    shot3 = ShotRecord(
        shot_id="s3",
        match_id=match_id,
        shot_number=3,
        hitter="PLAYER",
        shot_type="SMASH",
        speed_kmh=36.0,
        spin="NONE",
        landing_x=0.5,
        landing_z=1.1,
        result="WINNER",
    )
    stats.shots.extend([shot1, shot2, shot3])

    temp_storage.save_match(stats)

    # Verify matches list
    matches = temp_storage.get_matches()
    assert len(matches) == 1
    assert matches[0]["match_id"] == match_id
    assert matches[0]["winner"] == "PLAYER"
    assert matches[0]["player_score"] == 11

    # Verify shots list
    shots = temp_storage.get_shots_by_match(match_id)
    assert len(shots) == 3
    assert shots[2]["shot_type"] == "SMASH"
    assert shots[2]["result"] == "WINNER"

    # Verify lifetime aggregates
    agg = temp_storage.get_aggregated_stats()
    assert agg["total_matches"] == 1
    assert agg["player_wins"] == 1
    assert agg["win_rate_pct"] == 100.0
    assert agg["total_smashes"] == 3


def test_analytics_breakdown():
    shots = [
        {"hitter": "PLAYER", "shot_type": "FOREHAND", "speed_kmh": 20.0, "result": "IN_PLAY"},
        {"hitter": "AI", "shot_type": "BACKHAND", "speed_kmh": 22.0, "result": "IN_PLAY"},
        {"hitter": "PLAYER", "shot_type": "SMASH", "speed_kmh": 35.0, "result": "WINNER"},
        {"hitter": "PLAYER", "shot_type": "DRIVE", "speed_kmh": 15.0, "result": "OUT"},
    ]
    res = PingPongAnalytics.compute_match_breakdown(shots)
    assert res["total_shots"] == 4
    assert res["player_shots"] == 3
    assert res["player_unforced_errors"] == 1
    assert res["player_winners"] == 1
    assert res["max_speed"] == 35.0
