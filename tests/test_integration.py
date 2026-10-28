"""End-to-end integration test verifying full gameplay loop, rendering and telemetry."""

import numpy as np
import pytest

from src.engine.game_state import GameStateMachine, MatchStatus
from src.ui.renderer_3d import PerspectiveRenderer3D
from src.databricks.storage import LocalDeltaStorage
from src.databricks.analytics import PingPongAnalytics


def test_full_match_simulation_and_rendering(tmp_path):
    # 1. Initialize match
    game = GameStateMachine(difficulty="EASY")
    renderer = PerspectiveRenderer3D()
    db_file = tmp_path / "integration_matches.db"
    storage = LocalDeltaStorage(db_path=str(db_file))

    assert game.status == MatchStatus.READY

    # 2. Trigger Serve
    game.trigger_serve()
    assert game.status == MatchStatus.RALLY
    assert game.ball.in_play is True
    assert len(game.stats.shots) == 1

    # 3. Simulate 60 frames of play
    for _ in range(60):
        # AI and physics step
        game.update(dt=0.016)

        # Render 3D scene
        frame = renderer.render_frame(game)
        assert frame is not None
        assert frame.shape == (renderer.height, renderer.width, 3)

    # 4. Award a point to test state transition
    game._award_point("PLAYER", "WINNER POINT!")
    assert game.player_score == 1
    # After awarding point, game resets for serve
    assert game.status == MatchStatus.SERVING

    # 5. Fast-forward to 10-9, then award winning point for 11-9 game over
    game.player_score = 10
    game.ai_score = 9
    game._award_point("PLAYER", "CHAMPIONSHIP WINNER")
    assert game.status == MatchStatus.GAME_OVER
    assert game.stats.winner == "PLAYER"
    assert game.player_score == 11

    # 6. Save match to storage
    storage.save_match(game.stats)

    # 7. Validate telemetry persistence and analytics
    saved_matches = storage.get_matches()
    assert len(saved_matches) == 1
    assert saved_matches[0]["match_id"] == game.match_id
    assert saved_matches[0]["winner"] == "PLAYER"
    assert saved_matches[0]["player_score"] == 11

    shots = storage.get_shots_by_match(game.match_id)
    breakdown = PingPongAnalytics.compute_match_breakdown(shots)
    assert breakdown["total_shots"] >= 1
