"""Visual and sound effects manager for table tennis impacts and celebrations."""

import time
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class VisualSpark:
    x: int
    y: int
    radius: float
    max_radius: float
    color: Tuple[int, int, int]
    life: float  # 0.0 to 1.0


class SoundEffectsManager:
    """Manages hit sparks, impact waves and celebratory audio visual cues."""

    def __init__(self):
        self.sparks: List[VisualSpark] = []

    def trigger_hit_spark(self, screen_x: int, screen_y: int, shot_type: str = "DRIVE"):
        """Spawn visual impact spark on racket or table contact."""
        color = (0, 220, 255) if shot_type == "SMASH" else (255, 255, 255)
        max_r = 35.0 if shot_type == "SMASH" else 20.0
        self.sparks.append(VisualSpark(
            x=screen_x,
            y=screen_y,
            radius=4.0,
            max_radius=max_r,
            color=color,
            life=1.0,
        ))

    def update_and_render(self, frame: np.ndarray, dt: float):
        """Update spark lifetimes and render onto frame."""
        surviving = []
        for s in self.sparks:
            s.life -= dt * 3.5
            s.radius += (s.max_radius - s.radius) * 0.4
            if s.life > 0.05:
                alpha = s.life
                cv2.circle(
                    frame,
                    (s.x, s.y),
                    int(s.radius),
                    (int(s.color[0] * alpha), int(s.color[1] * alpha), int(s.color[2] * alpha)),
                    2,
                )
                surviving.append(s)
        self.sparks = surviving
