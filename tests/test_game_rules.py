"""Unit tests for ITTF rules, scoring state machine, and match lifecycle."""

from src.engine.game_state import GameStateMachine, MatchStatus


def test_game_initialization():
    sm = GameStateMachine(difficulty="MEDIUM")
    assert sm.player_score == 0
    assert sm.ai_score == 0
    assert sm.server == "PLAYER"
    assert sm.status == MatchStatus.READY

    sm.prepare_serve()
    assert sm.status == MatchStatus.SERVING


def test_serve_rotation():
    sm = GameStateMachine()
    # Award points to test rotation every 2 points
    sm._award_point("PLAYER", "Test Point 1")
    assert sm.player_score == 1
    assert sm.server == "PLAYER"  # Point 1 -> still Player serve

    sm._award_point("ROBERVAL", "Test Point 2")
    assert sm.ai_score == 1
    assert sm.player_score == 1
    assert sm.server == "ROBERVAL"  # Point 2 -> rotates to Roberval serve

    sm._award_point("PLAYER", "Test Point 3")
    assert sm.server == "ROBERVAL"  # Point 3 -> still Roberval serve

    sm._award_point("PLAYER", "Test Point 4")
    assert sm.server == "PLAYER"  # Point 4 -> rotates back to Player


def test_serve_mechanics_fist_ready_and_arm_strike():
    sm = GameStateMachine(difficulty="MEDIUM")
    assert sm.status == MatchStatus.READY

    # 1. Closing fist prepares the serve
    sm.prepare_serve()
    assert sm.status == MatchStatus.SERVING
    assert sm.ball.in_play is True
    assert sm.ball.vz == 0.0  # Floating waiting for arm strike

    # 2. Strike serve with arm swing speed
    sm.strike_serve(swing_speed=3.0, vx_offset=0.2, vy_lift=0.4)
    assert sm.status == MatchStatus.RALLY
    assert sm.ball.vz > 4.0  # Scaled by arm swing speed
    assert sm.ball.vy > 1.6  # Upward arc over net
    assert sm.ball.vx > 0.0  # Horizontal angle
    assert len(sm.stats.shots) == 1
    assert sm.stats.shots[0].shot_type == "SERVE"


def test_deuce_and_win_condition():
    sm = GameStateMachine()
    # Bring score to 10-10 (Deuce)
    sm.player_score = 10
    sm.ai_score = 10

    # 11-10 is NOT game over (needs 2 points lead)
    sm._award_point("PLAYER", "Point 11")
    assert sm.status != MatchStatus.GAME_OVER
    assert sm.player_score == 11

    # 12-10 IS game over!
    sm._award_point("PLAYER", "Point 12")
    assert sm.status == MatchStatus.GAME_OVER
    assert sm.stats.winner == "PLAYER"
    assert sm.stats.player_score == 12
    assert sm.stats.ai_score == 10
