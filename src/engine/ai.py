"""Computer Opponent AI for table tennis with trajectory prediction and difficulty tiers."""

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from src.engine.paddle import Paddle
from src.engine.physics import Ball


@dataclass
class AIDifficulty:
    """Parameters controlling AI reaction time, accuracy and speed."""
    name: str
    reaction_delay: float      # seconds before AI reacts to incoming shot
    max_paddle_speed: float    # m/s
    error_variance: float      # standard deviation of landing point error (meters)
    smash_probability: float   # probability of attempting a smash on high balls
    corner_aim_tendency: float # propensity to hit towards sharp table corners


DIFFICULTIES = {
    "EASY": AIDifficulty(
        name="EASY",
        reaction_delay=0.35,
        max_paddle_speed=1.3,
        error_variance=0.35,
        smash_probability=0.0,
        corner_aim_tendency=0.0,
    ),
    "MEDIUM": AIDifficulty(
        name="MEDIUM",
        reaction_delay=0.10,
        max_paddle_speed=3.6,
        error_variance=0.12,
        smash_probability=0.30,
        corner_aim_tendency=0.45,
    ),
    "HARD": AIDifficulty(
        name="HARD",
        reaction_delay=0.03,
        max_paddle_speed=5.8,
        error_variance=0.04,
        smash_probability=0.70,
        corner_aim_tendency=0.85,
    ),
    "CHAMPION": AIDifficulty(
        name="CHAMPION",
        reaction_delay=0.01,
        max_paddle_speed=7.5,
        error_variance=0.02,
        smash_probability=0.90,
        corner_aim_tendency=0.95,
    ),
}


class TableTennisAI:
    """Intelligent AI Opponent inspired by Wii Sports Ping Pong champions."""

    def __init__(self, difficulty: str = "MEDIUM"):
        self.difficulty: AIDifficulty = DIFFICULTIES.get(difficulty.upper(), DIFFICULTIES["MEDIUM"])
        self.target_x: float = 0.0
        self.target_y: float = 0.20
        self.target_z: float = 1.60
        self.last_predicted_time: float = 0.0
        self.rng = random.Random(42)

    def set_difficulty(self, difficulty: str):
        self.difficulty = DIFFICULTIES.get(difficulty.upper(), DIFFICULTIES["MEDIUM"])

    def predict_interception(self, ball: Ball) -> Tuple[float, float, float]:
        """Predict where ball will intersect AI's hitting plane (Z ~ 1.55)."""
        # If ball is heading away or stopped, return center resting spot near table baseline
        if ball.vz <= 0.2 or not ball.in_play:
            return 0.0, 0.18, 1.55

        z_strike = 1.55
        dz = z_strike - ball.z
        if dz <= 0:
            return ball.x, max(0.08, min(0.35, ball.y)), z_strike

        dt_intercept = dz / ball.vz

        # Ballistic projection for X and Y
        pred_x = ball.x + ball.vx * dt_intercept
        pred_y = ball.y + ball.vy * dt_intercept - 0.5 * 9.81 * (dt_intercept ** 2)

        # Handle table bounce estimation: realistic apex 12cm to 26cm above table
        if pred_y < 0.05:
            pred_y = min(0.26, max(0.12, 0.12 + abs(ball.vy) * 0.06))

        # Clamp to realistic player reach grounded near table surface
        pred_x = max(-0.85, min(0.85, pred_x))
        pred_y = max(0.08, min(0.35, pred_y))

        # Add difficulty-based stochastic noise
        err = self.difficulty.error_variance
        noisy_x = pred_x + self.rng.gauss(0, err)
        noisy_y = pred_y + self.rng.gauss(0, err * 0.5)

        return float(noisy_x), float(noisy_y), z_strike

    def update(self, ai_paddle: Paddle, ball: Ball, dt: float):
        """Update AI paddle movement and trigger hits."""
        if dt <= 0:
            dt = 1.0 / 60.0

        # Only actively hunt when ball is moving towards AI
        if ball.vz > 0.1 and ball.z > -0.2:
            tx, ty, tz = self.predict_interception(ball)
            self.target_x = tx
            self.target_y = ty
            self.target_z = tz
        else:
            # Return towards center resting stance
            self.target_x = 0.0
            self.target_y = 0.20
            self.target_z = 1.60

        # Move towards target with speed clamped by AI difficulty
        max_speed = self.difficulty.max_paddle_speed
        max_step = max_speed * dt

        dx = self.target_x - ai_paddle.x
        dy = self.target_y - ai_paddle.y
        dz = self.target_z - ai_paddle.z

        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > 1e-4:
            step = min(dist, max_step)
            new_x = ai_paddle.x + (dx / dist) * step
            new_y = ai_paddle.y + (dy / dist) * step
            new_z = ai_paddle.z + (dz / dist) * step
        else:
            new_x, new_y, new_z = self.target_x, self.target_y, self.target_z

        # Swing impulse when ball enters strike range
        if 1.20 <= ball.z <= 1.65 and ball.vz > 0:
            # Lift ball up and forward towards player with clean arc
            ai_paddle.vx = -0.3 if self.rng.random() < self.difficulty.corner_aim_tendency else 0.3
            ai_paddle.vy = 2.8
            ai_paddle.vz = -3.4
        else:
            ai_paddle.update_position(new_x, new_y, new_z, dt)
