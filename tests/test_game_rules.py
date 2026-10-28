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

    sm._award_point("AI", "Test Point 2")
    assert sm.ai_score == 1
    assert sm.player_score == 1
    assert sm.server == "AI"  # Point 2 -> rotates to AI serve

    sm._award_point("PLAYER", "Test Point 3")
    assert sm.server == "AI"  # Point 3 -> still AI serve

    sm._award_point("PLAYER", "Test Point 4")
    assert sm.server == "PLAYER"  # Point 4 -> rotates back to Player


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
