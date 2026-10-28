"""Gesture and stroke classifier for table tennis paddle swings."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class GestureResult:
    stroke_type: str         # "FOREHAND", "BACKHAND", "SMASH", "LOB", "DRIVE", "IDLE"
    swing_speed: float       # m/s
    swing_speed_kmh: float   # km/h
    is_active_swing: bool
    confidence: float


class GestureClassifier:
    """Classifies physical hand movements into table tennis shots."""

    def __init__(self, right_handed: bool = True):
        self.right_handed = right_handed
        self.swing_threshold = 2.2     # m/s threshold for deliberate stroke
        self.smash_threshold = 6.5     # m/s for power smash

    def classify(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray,
    ) -> GestureResult:
        """Analyze velocity and acceleration vectors to identify stroke type."""
        vx, vy, vz = velocity[0], velocity[1], velocity[2]
        speed = float(np.linalg.norm(velocity))
        speed_kmh = speed * 3.6

        if speed < self.swing_threshold:
            return GestureResult(
                stroke_type="IDLE",
                swing_speed=speed,
                swing_speed_kmh=speed_kmh,
                is_active_swing=False,
                confidence=0.85,
            )

        # Smash: high speed with downward or strong forward impulse
        if speed >= self.smash_threshold and (vy < -1.0 or vz > 3.5):
            return GestureResult(
                stroke_type="SMASH",
                swing_speed=speed,
                swing_speed_kmh=speed_kmh,
                is_active_swing=True,
                confidence=0.95,
            )

        # Lob: strong upward lift
        if vy > 3.0 and speed > 3.5:
            return GestureResult(
                stroke_type="LOB",
                swing_speed=speed,
                swing_speed_kmh=speed_kmh,
                is_active_swing=True,
                confidence=0.90,
            )

        # Forehand vs Backhand based on horizontal sweep
        # For right-handed player: rightward stroke is Forehand, leftward is Backhand
        if self.right_handed:
            if vx > 1.8:
                stroke = "FOREHAND"
            elif vx < -1.8:
                stroke = "BACKHAND"
            else:
                stroke = "DRIVE"
        else:
            if vx < -1.8:
                stroke = "FOREHAND"
            elif vx > 1.8:
                stroke = "BACKHAND"
            else:
                stroke = "DRIVE"

        return GestureResult(
            stroke_type=stroke,
            swing_speed=speed,
            swing_speed_kmh=speed_kmh,
            is_active_swing=True,
            confidence=0.88,
        )
