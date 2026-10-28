"""Paddle kinematics, collision detection, and momentum transfer."""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from src.config import PaddleConfig


@dataclass
class Paddle:
    """3D Virtual Table Tennis Paddle with Wii Sports Arcade Assistance."""
    is_player: bool
    config: PaddleConfig = field(default_factory=PaddleConfig)
    x: float = 0.0
    y: float = 0.20
    z: float = -1.40
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    tilt_x: float = 0.0  # radians
    tilt_y: float = 0.0  # radians
    radius: float = 0.28
    depth_tolerance: float = 0.35  # Generous depth tolerance
    assisted_mode: bool = True     # Wii Sports style magnetic assist

    def __post_init__(self):
        self.radius = self.config.RADIUS
        if not self.is_player:
            self.z = self.config.AI_Z_REST
        else:
            self.z = self.config.PLAYER_Z_REST

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def normal(self) -> np.ndarray:
        """Surface normal vector of paddle face pointing towards opponent side."""
        base_dir = 1.0 if self.is_player else -1.0
        # Include tilt
        nx = math.sin(self.tilt_y)
        ny = math.sin(self.tilt_x)
        nz = base_dir * math.cos(self.tilt_x) * math.cos(self.tilt_y)
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > 1e-6:
            return np.array([nx / norm, ny / norm, nz / norm], dtype=np.float64)
        return np.array([0.0, 0.0, base_dir], dtype=np.float64)

    def update_position(self, target_x: float, target_y: float, target_z: Optional[float], dt: float):
        """Update paddle position with smoothing and compute velocity."""
        if dt <= 0:
            dt = 1.0 / 60.0

        new_x = float(target_x)
        new_y = float(target_y)
        new_z = float(target_z) if target_z is not None else self.z

        # Velocity = displacement / dt (capped to physical limits)
        calc_vx = (new_x - self.x) / dt
        calc_vy = (new_y - self.y) / dt
        calc_vz = (new_z - self.z) / dt

        # Cap velocity to avoid crazy teleport spikes
        v_mag = math.sqrt(calc_vx**2 + calc_vy**2 + calc_vz**2)
        if v_mag > self.config.MAX_SWING_SPEED:
            scale = self.config.MAX_SWING_SPEED / v_mag
            calc_vx *= scale
            calc_vy *= scale
            calc_vz *= scale

        # Exponential smoothing on velocity
        alpha = 0.6
        self.vx = alpha * calc_vx + (1.0 - alpha) * self.vx
        self.vy = alpha * calc_vy + (1.0 - alpha) * self.vy
        self.vz = alpha * calc_vz + (1.0 - alpha) * self.vz

        self.x = new_x
        self.y = new_y
        self.z = new_z

    def check_ball_collision(
        self,
        ball_pos: np.ndarray,
        ball_radius: float,
    ) -> Tuple[bool, float, float]:
        """
        Check if ball is colliding with the paddle blade (with arcade assistance).
        Returns:
            (collided, dist_to_center, z_dist)
        """
        bx, by, bz = ball_pos
        dx = bx - self.x
        dy = by - self.y
        radial_dist = math.sqrt(dx * dx + dy * dy)
        z_dist = abs(bz - self.z)

        # Base physical blade hit
        if radial_dist <= (self.radius + ball_radius):
            if z_dist <= (self.depth_tolerance + ball_radius):
                return True, radial_dist, z_dist

        # Arcade Assisted Mode (Wii Sports style) for player
        if self.assisted_mode and self.is_player:
            # When ball is in player strike zone (Z between -1.85 and -1.15)
            if -1.85 <= bz <= -1.15:
                # If paddle is within assisted reach
                if radial_dist <= self.config.ASSISTED_RADIUS and z_dist <= (self.depth_tolerance + 0.15):
                    return True, radial_dist, z_dist

        return False, radial_dist, z_dist

    def compute_rebound_velocity(
        self,
        ball_velocity: np.ndarray,
        ball_pos: np.ndarray,
        restitution: float = 0.90,
    ) -> Tuple[np.ndarray, str]:
        """
        Compute new ball velocity after striking the paddle with consistent returns.
        """
        # Direction to opponent side
        forward_dir = 1.0 if self.is_player else -1.0

        # Base rebound calculation
        v_rel = ball_velocity - self.velocity
        n = self.normal
        v_rel_n = np.dot(v_rel, n)

        if (self.is_player and v_rel_n < 0) or (not self.is_player and v_rel_n > 0):
            v_rebound = ball_velocity - (1.0 + restitution) * v_rel_n * n
            v_rebound += 0.8 * self.velocity
        else:
            v_rebound = ball_velocity + 0.8 * self.velocity

        # Arc stabilization: Ensure ball clears the net comfortably
        forward_speed = max(3.0, min(6.5, abs(v_rebound[2]) + abs(self.vz) * 0.5))
        v_rebound[2] = forward_speed * forward_dir

        # High arc so user can follow ball easily
        v_rebound[1] = max(1.6, min(3.0, abs(v_rebound[1]) * 0.7 + 1.2))

        # Angle slightly toward table center or cross-court based on swing
        if self.is_player:
            # Paddle lateral offset influences return angle
            offset_x = (ball_pos[0] - self.x)
            v_rebound[0] = np.clip(self.vx * 0.6 + offset_x * 2.0, -1.8, 1.8)

        # Shot classification
        swing_speed = self.speed
        if swing_speed > 7.0 or self.vz > 4.0:
            shot_type = "SMASH"
            v_rebound[2] = 5.8 * forward_dir
            v_rebound[1] = 1.2
        elif self.vx > 1.8:
            shot_type = "FOREHAND"
        elif self.vx < -1.8:
            shot_type = "BACKHAND"
        else:
            shot_type = "DRIVE"

        return v_rebound, shot_type
