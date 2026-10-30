"""Visual and sound effects manager for table tennis impacts and celebrations."""

import math
import random
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
    life: float  # 1.0 down to 0.0


@dataclass
class ParticleShard:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float  # 1.0 down to 0.0
    size: float


class SoundEffectsManager:
    """Manages spatial 3D audio, hit sparks, impact shockwaves and celebratory visual cues."""

    def __init__(self):
        self.sparks: List[VisualSpark] = []
        self.shards: List[ParticleShard] = []
        self._audio_available = False
        self._hit_pool = []
        self._smash_sound = None
        self._bounce_sound = None
        self._net_sound = None
        self._cheer_sound = None
        self._point_sound = None
        self._init_audio()

    def _init_audio(self):
        """Initialize pygame audio mixer and synthesize layered spatial sound waves."""
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

            sr = 22050

            # 1. Hit Sound Variations (paddle contact with slight pitch offsets)
            for freq in (425, 445, 465):
                t_hit = np.linspace(0, 0.085, int(sr * 0.085), False)
                w_hit = (np.sin(2 * np.pi * freq * t_hit) * 0.7 +
                         np.sin(2 * np.pi * (freq * 1.5) * t_hit) * 0.3) * np.exp(-t_hit * 38) * 0.45
                mono_hit = (w_hit * 32767).astype(np.int16)
                stereo_hit = np.column_stack((mono_hit, mono_hit))
                self._hit_pool.append(pygame.sndarray.make_sound(stereo_hit))

            # 2. Smash Sound (Punchy layered crack + low sub kick)
            t_smash = np.linspace(0, 0.16, int(sr * 0.16), False)
            # Crack layer (sawtooth approximation)
            w_crack = np.sin(2 * np.pi * 640 * t_smash) * np.exp(-t_smash * 28) * 0.55
            # Sub-kick thump layer (85Hz decaying)
            w_kick = np.sin(2 * np.pi * 85 * t_smash) * np.exp(-t_smash * 20) * 0.65
            mono_smash = ((w_crack + w_kick) * 0.55 * 32767).astype(np.int16)
            stereo_smash = np.column_stack((mono_smash, mono_smash))
            self._smash_sound = pygame.sndarray.make_sound(stereo_smash)

            # 3. Bounce Sound (Crisp wooden ping on ITTF tabletop)
            t_bnc = np.linspace(0, 0.065, int(sr * 0.065), False)
            w_bnc = (np.sin(2 * np.pi * 290 * t_bnc) * 0.8 +
                     np.sin(2 * np.pi * 580 * t_bnc) * 0.2) * np.exp(-t_bnc * 42) * 0.35
            mono_bnc = (w_bnc * 32767).astype(np.int16)
            stereo_bnc = np.column_stack((mono_bnc, mono_bnc))
            self._bounce_sound = pygame.sndarray.make_sound(stereo_bnc)

            # 4. Net Scrape Sound
            t_net = np.linspace(0, 0.08, int(sr * 0.08), False)
            noise = np.random.uniform(-1, 1, len(t_net)) * 0.4
            w_net = (np.sin(2 * np.pi * 180 * t_net) * 0.6 + noise) * np.exp(-t_net * 25) * 0.3
            mono_net = (w_net * 32767).astype(np.int16)
            stereo_net = np.column_stack((mono_net, mono_net))
            self._net_sound = pygame.sndarray.make_sound(stereo_net)

            # 5. Crowd Cheer / Applause Burst
            t_cheer = np.linspace(0, 0.6, int(sr * 0.6), False)
            crowd = np.random.uniform(-1, 1, len(t_cheer))
            env = np.sin(np.pi * (t_cheer / 0.6)) ** 2 * 0.35
            mono_cheer = (crowd * env * 32767).astype(np.int16)
            stereo_cheer = np.column_stack((mono_cheer, mono_cheer))
            self._cheer_sound = pygame.sndarray.make_sound(stereo_cheer)

            # 6. Point Scored Melodic Chime (587Hz -> 880Hz harmonic ding)
            t_pt = np.linspace(0, 0.32, int(sr * 0.32), False)
            split_idx = int(len(t_pt) * 0.45)
            w_pt1 = np.sin(2 * np.pi * 587 * t_pt[:split_idx]) * np.exp(-t_pt[:split_idx] * 8) * 0.35
            t_rem = t_pt[split_idx:] - t_pt[split_idx]
            w_pt2 = np.sin(2 * np.pi * 880 * t_rem) * np.exp(-t_rem * 7) * 0.42
            w_point = np.concatenate((w_pt1, w_pt2))
            mono_pt = (w_point * 32767).astype(np.int16)
            stereo_pt = np.column_stack((mono_pt, mono_pt))
            self._point_sound = pygame.sndarray.make_sound(stereo_pt)

            self._audio_available = True
        except Exception:
            self._audio_available = False

    def _apply_spatial_pan(self, sound, ball_x: float = 0.0):
        """Play sound with 3D stereo panning (audio-design skill)."""
        if not sound:
            return
        try:
            # Map ball X (-0.76 to 0.76) to pan (-1.0 left to +1.0 right)
            pan = max(-1.0, min(1.0, ball_x / 0.76))
            left = math.sqrt(0.5 * (1.0 - pan))
            right = math.sqrt(0.5 * (1.0 + pan))
            ch = sound.play()
            if ch:
                ch.set_volume(max(0.15, min(1.0, left)), max(0.15, min(1.0, right)))
        except Exception:
            try:
                sound.play()
            except Exception:
                pass

    def play_hit(self, is_smash: bool = False, ball_x: float = 0.0):
        """Play paddle contact sound with stereo panning and pitch variation."""
        if not self._audio_available:
            return
        if is_smash and self._smash_sound:
            self._apply_spatial_pan(self._smash_sound, ball_x)
        elif self._hit_pool:
            sound = random.choice(self._hit_pool)
            self._apply_spatial_pan(sound, ball_x)

    def play_bounce(self, ball_x: float = 0.0):
        """Play table bounce sound with spatial panning."""
        if not self._audio_available:
            return
        if self._bounce_sound:
            self._apply_spatial_pan(self._bounce_sound, ball_x)

    def play_net(self):
        """Play net tape clip scrape."""
        if not self._audio_available:
            return
        if self._net_sound:
            self._net_sound.play()

    def play_cheer(self):
        """Play crowd celebration cheer for rally milestones."""
        if not self._audio_available:
            return
        if self._cheer_sound:
            self._cheer_sound.play()

    def play_point_scored(self, winner: str = "PLAYER"):
        """Play celebratory chime on point won."""
        if not self._audio_available:
            return
        try:
            if self._point_sound:
                self._point_sound.play()
            if winner == "PLAYER" and self._cheer_sound:
                self._cheer_sound.set_volume(0.35)
                self._cheer_sound.play()
        except Exception:
            pass

    def trigger_hit_spark(self, screen_x: int, screen_y: int, shot_type: str = "DRIVE", ball_x: float = 0.0):
        """Spawn expanding shockwave ring and particle shards (game-feel juice)."""
        is_smash = ("SMASH" in shot_type)
        self.play_hit(is_smash=is_smash, ball_x=ball_x)

        # Expanding shockwave ring
        ring_color = (0, 230, 255) if is_smash else (255, 255, 255)
        max_r = 45.0 if is_smash else 24.0
        self.sparks.append(VisualSpark(
            x=screen_x,
            y=screen_y,
            radius=4.0,
            max_radius=max_r,
            color=ring_color,
            life=1.0,
        ))

        # Flying particle shards
        shard_count = 14 if is_smash else 7
        for _ in range(shard_count):
            angle = random.uniform(0, 2 * math.pi)
            spd = random.uniform(2.5, 7.0) if is_smash else random.uniform(1.5, 4.0)
            shard_color = (0, 180, 255) if is_smash else (255, 240, 180)
            self.shards.append(ParticleShard(
                x=float(screen_x),
                y=float(screen_y),
                vx=math.cos(angle) * spd,
                vy=math.sin(angle) * spd - 1.2,
                color=shard_color,
                life=1.0,
                size=random.uniform(2.0, 4.0),
            ))

    def update_and_render(self, frame: np.ndarray, dt: float):
        """Update and render visual sparks and particles with alpha decay."""
        # 1. Render Shockwave Rings
        surviving_sparks = []
        for s in self.sparks:
            s.life -= dt * 3.8
            s.radius += (s.max_radius - s.radius) * 0.45
            if s.life > 0.05:
                alpha = max(0.0, min(1.0, s.life))
                c = (int(s.color[0] * alpha), int(s.color[1] * alpha), int(s.color[2] * alpha))
                cv2.circle(frame, (s.x, s.y), int(s.radius), c, 2)
                surviving_sparks.append(s)
        self.sparks = surviving_sparks

        # 2. Render Particle Shards
        surviving_shards = []
        for p in self.shards:
            p.life -= dt * 3.2
            p.x += p.vx
            p.y += p.vy
            p.vy += 6.0 * dt  # Gravity
            if p.life > 0.05:
                alpha = max(0.0, min(1.0, p.life))
                c = (int(p.color[0] * alpha), int(p.color[1] * alpha), int(p.color[2] * alpha))
                cv2.circle(frame, (int(p.x), int(p.y)), max(1, int(p.size * alpha)), c, -1)
                surviving_shards.append(p)
        self.shards = surviving_shards
