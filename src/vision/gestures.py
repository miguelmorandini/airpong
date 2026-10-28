"""Gesture and stroke classifier for table tennis paddle swings."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class GestureResult:
    stroke_type: str         # "GANCHO_FOREHAND", "GANCHO_BACKHAND", "GANCHO_SMASH", "DRIVE", "IDLE"
    swing_speed: float       # m/s
    swing_speed_kmh: float   # km/h
    is_active_swing: bool
    is_hook_swing: bool      # True when a dynamic hook motion (gancho) is detected
    hook_power: float        # Power multiplier (1.0x to 2.5x)
    confidence: float


class GestureClassifier:
    """Classifies physical hand and arm movements into table tennis hook strokes."""

    def __init__(self, right_handed: bool = True):
        self.right_handed = right_handed
        # Natural threshold for fluid arm/hand hook swing
        self.swing_threshold = 1.15    # m/s threshold for deliberate hook motion
        self.smash_threshold = 3.6     # m/s for power smash hook

    def classify(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray,
    ) -> GestureResult:
        """Analyze velocity and acceleration vectors to identify arm/hand hook stroke."""
        vx, vy, vz = velocity[0], velocity[1], velocity[2]
        speed = float(np.linalg.norm(velocity))
        speed_kmh = speed * 3.6

        if speed < self.swing_threshold:
            return GestureResult(
                stroke_type="IDLE",
                swing_speed=speed,
                swing_speed_kmh=speed_kmh,
                is_active_swing=False,
                is_hook_swing=False,
                hook_power=1.0,
                confidence=0.85,
            )

        # Power calculation based on swing velocity
        hook_power = max(1.0, min(2.5, 1.0 + (speed - self.swing_threshold) * 0.45))

        # 1. Smash Hook: downward or strong forward snap
        if speed >= self.smash_threshold and (vy < -0.8 or vz > 2.0):
            return GestureResult(
                stroke_type="GANCHO_SMASH",
                swing_speed=speed,
                swing_speed_kmh=speed_kmh,
                is_active_swing=True,
                is_hook_swing=True,
                hook_power=hook_power * 1.25,
                confidence=0.95,
            )

        # 2. Forehand Hook vs Backhand Hook based on lateral arm sweep
        # Right-handed: positive vx (sweep to right/inward) is Forehand Hook, negative vx is Backhand Hook
        if self.right_handed:
            if vx > 0.8:
                stroke = "GANCHO_FOREHAND"
            elif vx < -0.8:
                stroke = "GANCHO_BACKHAND"
            else:
                stroke = "DRIVE"
        else:
            if vx < -0.8:
                stroke = "GANCHO_FOREHAND"
            elif vx > 0.8:
                stroke = "GANCHO_BACKHAND"
            else:
                stroke = "DRIVE"

        return GestureResult(
            stroke_type=stroke,
            swing_speed=speed,
            swing_speed_kmh=speed_kmh,
            is_active_swing=True,
            is_hook_swing=True,
            hook_power=hook_power,
            confidence=0.90,
        )
