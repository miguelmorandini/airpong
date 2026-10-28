"""Unit tests for 3D physics engine and table boundaries."""

import numpy as np
import pytest

from src.engine.table import Table
from src.engine.paddle import Paddle
from src.engine.physics import Ball, PhysicsEngine


def test_table_boundaries():
    table = Table()
    # Center of table
    assert table.is_on_table(0.0, 0.0) is True
    # Corners
    assert table.is_on_table(table.half_width, table.half_length) is True
    assert table.is_on_table(-table.half_width, -table.half_length) is True
    # Outside table
    assert table.is_on_table(table.half_width + 0.1, 0.0) is False
    assert table.is_on_table(0.0, table.half_length + 0.1) is False

    # Player vs AI sides
    assert table.is_player_side(-0.5) is True
    assert table.is_ai_side(0.5) is True
    assert table.is_player_side(0.5) is False
    assert table.is_ai_side(-0.5) is False


def test_net_collision_detection():
    table = Table()
    ball_radius = 0.02
    # Ball crossing from negative Z to positive Z below net height
    crossed = table.check_net_collision(
        z_prev=-0.05,
        z_curr=0.05,
        x=0.0,
        y=0.10,  # Below net height 0.1525m
        ball_radius=ball_radius,
    )
    assert crossed is True

    # Ball crossing above net height
    high_crossed = table.check_net_collision(
        z_prev=-0.05,
        z_curr=0.05,
        x=0.0,
        y=0.25,  # Well above net
        ball_radius=ball_radius,
    )
    assert high_crossed is False


def test_paddle_collision():
    paddle = Paddle(is_player=True)
    paddle.x = 0.0
    paddle.y = 0.20
    paddle.z = -1.65

    # Ball right on paddle center
    ball_pos = np.array([0.0, 0.20, -1.65])
    hit, dist, z_dist = paddle.check_ball_collision(ball_pos, ball_radius=0.02)
    assert hit is True
    assert dist == pytest.approx(0.0, abs=1e-3)

    # Ball far away
    far_ball = np.array([0.8, 0.20, -1.65])
    hit_far, _, _ = paddle.check_ball_collision(far_ball, ball_radius=0.02)
    assert hit_far is False


def test_ball_bounce_on_table():
    engine = PhysicsEngine()
    ball = Ball()
    ball.x = 0.0
    ball.z = -0.5  # Player side of table
    ball.y = 0.05
    ball.vy = -2.0  # Falling down
    ball.vx = 0.0
    ball.vz = 1.0

    player_p = Paddle(is_player=True)
    ai_p = Paddle(is_player=False)

    events = engine.step(ball, player_p, ai_p, dt=0.05)
    bounce_events = [e for e in events if e.event_type == "BOUNCE"]
    assert len(bounce_events) >= 1
    assert bounce_events[0].details == "PLAYER"
    # Velocity vy reversed (upwards)
    assert ball.vy > 0.0


def test_player_paddle_hit_rebatida_physics():
    """Verify calibrated return physics on player hits (no excessive float/ballooning)."""
    engine = PhysicsEngine()
    player_p = Paddle(is_player=True)
    player_p.x = 0.0
    player_p.y = 0.25
    player_p.z = -1.40
    player_p.vx = 1.0  # Hook forehand swing
    player_p.vy = 0.2
    player_p.vz = 0.5
    ai_p = Paddle(is_player=False)

    ball = Ball()
    ball.x = 0.05
    ball.y = 0.25
    ball.z = -1.35
    ball.vx = 0.0
    ball.vy = -0.5
    ball.vz = -3.0  # Incoming from opponent

    events = engine.step(ball, player_p, ai_p, dt=0.016)
    hit_events = [e for e in events if e.event_type == "HIT"]
    assert len(hit_events) == 1
    assert hit_events[0].hitter == "PLAYER"
    assert "GANCHO" in hit_events[0].details

    # Rebatida physics verification
    assert 3.2 <= ball.vz <= 5.8       # Clean, punchy forward drive
    assert 1.2 <= ball.vy <= 2.0       # Low, tight arc over net (no balloon lob)
    assert -1.0 <= ball.vx <= 1.0      # Contained lateral angle


def test_roberval_return_rebatida_physics():
    """Verify Roberval returns with natural, playable arc towards player."""
    engine = PhysicsEngine()
    player_p = Paddle(is_player=True)
    ai_p = Paddle(is_player=False)
    ai_p.x = 0.0
    ai_p.y = 0.25
    ai_p.z = 1.40

    ball = Ball()
    ball.x = 0.05
    ball.y = 0.25
    ball.z = 1.35
    ball.vx = 0.0
    ball.vy = -0.5
    ball.vz = 3.5  # Incoming from player

    events = engine.step(ball, player_p, ai_p, dt=0.016)
    hit_events = [e for e in events if e.event_type == "HIT"]
    assert len(hit_events) == 1
    assert hit_events[0].hitter == "ROBERVAL"

    # Roberval return physics verification
    assert -4.2 <= ball.vz <= -3.0     # Reaches player baseline
    assert 1.4 <= ball.vy <= 2.0       # Natural table tennis arc (no moonball)
    assert -0.8 <= ball.vx <= 0.8      # Lands on player table
