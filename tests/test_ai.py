"""Unit tests for AI Opponent behavior and trajectory prediction."""

from src.engine.ai import TableTennisAI, DIFFICULTIES
from src.engine.paddle import Paddle
from src.engine.physics import Ball


def test_ai_difficulties():
    for diff in ["EASY", "MEDIUM", "HARD", "CHAMPION"]:
        ai = TableTennisAI(difficulty=diff)
        assert ai.difficulty.name == diff
        assert ai.difficulty.max_paddle_speed > 0


def test_ai_predict_interception():
    ai = TableTennisAI(difficulty="CHAMPION")
    ball = Ball()
    ball.x = 0.1
    ball.y = 0.2
    ball.z = 0.0
    ball.vx = 0.0
    ball.vy = 0.5
    ball.vz = 4.0  # Moving towards AI

    tx, ty, tz = ai.predict_interception(ball)
    assert tz == 1.55
    # Since ball vx is 0 and noise in CHAMPION is tiny (0.02), tx should be close to 0.1
    assert abs(tx - 0.1) < 0.15


def test_ai_paddle_update_movement():
    ai = TableTennisAI(difficulty="MEDIUM")
    ai_paddle = Paddle(is_player=False)
    ai_paddle.x = 0.0
    ai_paddle.y = 0.2
    ai_paddle.z = 1.60

    ball = Ball()
    ball.x = 0.5
    ball.y = 0.2
    ball.z = 0.5
    ball.vz = 3.0

    # Advance AI
    ai.update(ai_paddle, ball, dt=0.1)
    # AI paddle should have moved towards positive X
    assert ai_paddle.x > 0.0
