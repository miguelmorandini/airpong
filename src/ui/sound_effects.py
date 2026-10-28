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
        self._audio_available = False
        self._hit_sound = None
        self._smash_sound = None
        self._bounce_sound = None
        self._init_audio()

    def _init_audio(self):
        """Initialize pygame audio mixer and synthesize impact sound waves."""
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2)

            sr = 22050
            # Hit sound (paddle thwack ~ 450Hz decaying)
            t_hit = np.linspace(0, 0.09, int(sr * 0.09), False)
            w_hit = np.sin(2 * np.pi * 440 * t_hit) * np.exp(-t_hit * 35) * 0.4
            mono_hit = (w_hit * 32767).astype(np.int16)
            self._hit_sound = pygame.sndarray.make_sound(np.column_stack((mono_hit, mono_hit)))

            # Smash sound (louder, punchier ~ 620Hz)
            t_smash = np.linspace(0, 0.14, int(sr * 0.14), False)
            w_smash = np.sin(2 * np.pi * 600 * t_smash) * np.exp(-t_smash * 25) * 0.6
            mono_smash = (w_smash * 32767).astype(np.int16)
            self._smash_sound = pygame.sndarray.make_sound(np.column_stack((mono_smash, mono_smash)))

            # Bounce sound (table tok ~ 280Hz)
            t_bnc = np.linspace(0, 0.06, int(sr * 0.06), False)
            w_bnc = np.sin(2 * np.pi * 280 * t_bnc) * np.exp(-t_bnc * 45) * 0.3
            mono_bnc = (w_bnc * 32767).astype(np.int16)
            self._bounce_sound = pygame.sndarray.make_sound(np.column_stack((mono_bnc, mono_bnc)))

            self._audio_available = True
        except Exception:
            self._audio_available = False

    def play_hit(self, is_smash: bool = False):
        """Play paddle contact sound."""
        if not self._audio_available:
            return
        try:
            if is_smash and self._smash_sound:
                self._smash_sound.play()
            elif self._hit_sound:
                self._hit_sound.play()
        except Exception:
            pass

    def play_bounce(self):
        """Play table bounce sound."""
        if not self._audio_available:
            return
        try:
            if self._bounce_sound:
                self._bounce_sound.play()
        except Exception:
            pass

    def trigger_hit_spark(self, screen_x: int, screen_y: int, shot_type: str = "DRIVE"):
        """Spawn visual impact spark on racket or table contact and play sound."""
        is_smash = (shot_type == "SMASH")
        self.play_hit(is_smash=is_smash)
        color = (0, 220, 255) if is_smash else (255, 255, 255)
        max_r = 35.0 if is_smash else 20.0
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
