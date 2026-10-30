"""3D Ball Physics simulation with drag, Magnus effect (spin), table bounce and net collisions."""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from src.config import BallConfig
from src.engine.table import Table
from src.engine.paddle import Paddle


@dataclass
class PhysicsEvent:
    """Event emitted during physics step (hit, bounce, net collision, fault)."""
    event_type: str  # "HIT", "BOUNCE", "NET", "FAULT", "OUT"
    hitter: Optional[str] = None       # "PLAYER" or "AI"
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_kmh: float = 0.0
    details: str = ""


@dataclass
class Ball:
    """3D Table Tennis Ball."""
    config: BallConfig = field(default_factory=BallConfig)
    x: float = 0.0
    y: float = 0.25
    z: float = -1.45
    vx: float = 0.0
    vy: float = 1.0
    vz: float = 4.0
    spin_x: float = 0.0   # rad/s - Topspin / Backspin
    spin_y: float = 0.0   # rad/s - Sidespin
    spin_z: float = 0.0   # rad/s - Corkscrew
    in_play: bool = True
    last_hitter: Optional[str] = None
    bounce_count_current_hit: int = 0
    bounced_player_side: bool = False
    bounced_ai_side: bool = False
    is_serve: bool = True
    squash: float = 0.0  # Elastic deformation on impacts (game-feel juice)
    trail: List[Tuple[float, float, float]] = field(default_factory=list)

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
    def speed_kmh(self) -> float:
        return self.speed * 3.6

    @property
    def radius(self) -> float:
        return self.config.RADIUS


class PhysicsEngine:
    """Simulates 3D physics of table tennis according to ITTF real-world mechanics."""

    def __init__(self, table: Optional[Table] = None, ball_config: Optional[BallConfig] = None):
        self.table = table or Table()
        self.ball_config = ball_config or BallConfig()
        self.cross_sectional_area = math.pi * (self.ball_config.RADIUS ** 2)

    def reset_serve(self, ball: Ball, server: str = "PLAYER"):
        """Prepare ball for serve from server's side."""
        ball.in_play = True
        ball.is_serve = True
        ball.last_hitter = None
        ball.bounce_count_current_hit = 0
        ball.bounced_player_side = False
        ball.bounced_ai_side = False
        ball.spin_x = 0.0
        ball.spin_y = 0.0
        ball.spin_z = 0.0
        ball.trail.clear()

        if server == "PLAYER":
            ball.x = 0.15
            ball.y = 0.28
            ball.z = -1.35
            ball.vx = 0.0
            ball.vy = 0.0
            ball.vz = 0.0
        else:
            ball.x = -0.15
            ball.y = 0.28
            ball.z = 1.35
            ball.vx = 0.0
            ball.vy = 0.0
            ball.vz = 0.0

    def step(
        self,
        ball: Ball,
        player_paddle: Paddle,
        ai_paddle: Paddle,
        dt: float = 1.0 / 60.0,
    ) -> List[PhysicsEvent]:
        """Advance simulation by dt seconds and resolve all collisions."""
        events: List[PhysicsEvent] = []
        if not ball.in_play:
            return events

        # Sub-stepping for ultra-high speed accuracy and tunnel prevention
        substeps = 4
        sub_dt = dt / substeps

        for _ in range(substeps):
            if not ball.in_play:
                break
            step_events = self._substep(ball, player_paddle, ai_paddle, sub_dt)
            events.extend(step_events)

        # Update motion trail
        ball.trail.append((ball.x, ball.y, ball.z))
        if len(ball.trail) > 12:
            ball.trail.pop(0)

        return events

    def _substep(
        self,
        ball: Ball,
        player_paddle: Paddle,
        ai_paddle: Paddle,
        dt: float,
    ) -> List[PhysicsEvent]:
        events: List[PhysicsEvent] = []
        prev_x, prev_y, prev_z = ball.x, ball.y, ball.z

        # 1. Aerodynamic Forces (Gravity + Drag + Magnus Spin)
        v = ball.velocity
        speed = float(np.linalg.norm(v))

        # Gravity
        accel = np.array([0.0, -self.ball_config.GRAVITY, 0.0], dtype=np.float64)

        if speed > 1e-4:
            # Air resistance (quadratic drag: F_drag = 0.5 * rho * Cd * A * v^2)
            f_drag = 0.5 * self.ball_config.AIR_DENSITY * self.ball_config.DRAG_COEFFICIENT * \
                     self.cross_sectional_area * speed
            a_drag = -(f_drag / self.ball_config.MASS) * (v / speed)
            accel += a_drag

            # Magnus effect: F_magnus = S * (omega x v)
            # Calibrated for vivid, satisfying curve shots ("dar efeito")
            spin_vec = np.array([ball.spin_x, ball.spin_y, ball.spin_z], dtype=np.float64)
            omega_cross_v = np.cross(spin_vec, v)
            k_magnus = 0.00065 / self.ball_config.MASS
            accel += k_magnus * omega_cross_v

        # Integrate velocity and position (Velocity Verlet / Symplectic Euler)
        ball.vx += accel[0] * dt
        ball.vy += accel[1] * dt
        ball.vz += accel[2] * dt

        ball.x += ball.vx * dt
        ball.y += ball.vy * dt
        ball.z += ball.vz * dt

        # Elastic squash decay (game-feel juice)
        if ball.squash > 0.0:
            ball.squash = max(0.0, ball.squash - dt * 6.0)

        # 2. Check Paddle Collisions with Motion Attack Model (Wii Sports / Arcade Style)
        # Player Paddle: Ball traveling towards player (vz < 0)
        is_crossing_player = (prev_z >= player_paddle.z and ball.z <= player_paddle.z)
        is_near_player_plane = abs(ball.z - player_paddle.z) <= (self.ball_config.RADIUS + 0.06)

        # Player strike arrival corridor in Z (from bounce near net to baseline)
        in_strike_window = (-1.80 <= ball.z <= -0.75)

        # Detect active attack motion from arm/hand kinematics, hook swing, gesture or buffered click
        arm_power = getattr(player_paddle, "arm_power", 1.0)
        hand_spd = getattr(player_paddle, "hand_speed", 0.0)
        p_speed = player_paddle.speed
        stroke_name = getattr(player_paddle, "stroke_type", None) or ""
        kinetic_phase = getattr(player_paddle, "kinetic_phase", "IDLE")
        attack_flag = getattr(player_paddle, "attack_triggered", False)

        is_attack_motion = (
            attack_flag
            or getattr(player_paddle, "is_hook_swing", False)
            or (p_speed >= 0.45)
            or (hand_spd >= 0.50)
            or (arm_power >= 1.25 and p_speed >= 0.25)
            or (stroke_name not in ("", "IDLE"))
            or (kinetic_phase in ("SNAP", "ACCELERATION", "WHIP"))
        )

        should_check_player = (
            ball.vz < 0.0
            and (
                (is_attack_motion and in_strike_window)
                or is_crossing_player
                or (is_near_player_plane and ball.z <= -1.15)
            )
        )

        if should_check_player:
            # Continuous Collision Detection: find exact intersection with player strike plane
            if is_crossing_player and abs(ball.z - prev_z) > 1e-6:
                t_hit = float(np.clip((player_paddle.z - prev_z) / (ball.z - prev_z), 0.0, 1.0))
                hit_x = float(prev_x + t_hit * (ball.x - prev_x))
                hit_y = float(prev_y + t_hit * (ball.y - prev_y))
                hit_z = float(player_paddle.z)
            else:
                hit_x = float(ball.x)
                hit_y = float(ball.y)
                hit_z = float(ball.z)

            dx = hit_x - player_paddle.x
            dy = hit_y - player_paddle.y
            radial_dist = math.sqrt(dx * dx + dy * dy)

            # In Motion Attack Mode, the motion/swing connects anywhere in table strike corridor
            if is_attack_motion:
                max_reach = 1.15  # Full table corridor reach
            else:
                vis_r = getattr(player_paddle.config, "VISUAL_RADIUS", 0.076)
                is_corner_shot = abs(hit_x) >= 0.40
                base_reach = getattr(player_paddle.config, "ASSISTED_RADIUS", 0.48)
                corner_assist = (vis_r * 0.40) if is_corner_shot else 0.0
                max_reach = base_reach + corner_assist

            # Strike zone: ball intercepted by attack motion or paddle face
            if radial_dist <= max_reach:
                # Magnetically align paddle cleanly to the ball
                player_paddle.x = hit_x
                player_paddle.y = hit_y
                player_paddle.z = hit_z - self.ball_config.RADIUS - 0.02
                if hasattr(player_paddle, "attack_triggered"):
                    player_paddle.attack_triggered = False

                contact_offset_x = dx
                p_norm = player_paddle.normal
                sweet_mult, sweet_tier = player_paddle.get_sweet_spot_tier(min(radial_dist, 0.045))

                swing_power = max(
                    getattr(player_paddle, "hook_power", 1.0),
                    player_paddle.speed * arm_power,
                    getattr(player_paddle, "hand_speed", 0.0),
                    1.1 if is_attack_motion else 1.0,
                )

                is_smash = (
                    ("SMASH" in stroke_name)
                    or (swing_power >= 2.2 and (player_paddle.vy < -0.15 or abs(player_paddle.vx) >= 0.8))
                    or (player_paddle.speed >= 2.6)
                    or (arm_power >= 1.45 and player_paddle.vy < -0.25)
                )

                # Timing angle control (Wii Sports mechanic):
                # Hitting early (in front) pulls cross-court; hitting late (deep) pushes down-the-line
                ideal_z = -1.35
                z_timing_diff = ideal_z - hit_z  # positive if struck in front of ideal line
                is_right = getattr(player_paddle, "right_handed", True)
                timing_angle = float(np.clip((-z_timing_diff * 0.40) if is_right else (z_timing_diff * 0.40), -0.35, 0.35))

                blade_pitch = getattr(player_paddle, "pitch", 0.0) if player_paddle.hand_detected else 0.0

                if is_smash:
                    shot_type = "GANCHO SMASH"
                    forward_speed = min(5.6, (4.6 + swing_power * 0.16) * sweet_mult)
                    vertical_lift = 1.65 + p_norm[1] * 0.35
                    ball.spin_x = 5.5
                elif player_paddle.vy < -0.30 or blade_pitch > 0.22 or "SLICE" in stroke_name or "CHOP" in stroke_name:
                    shot_type = "CORTE SLICE"
                    forward_speed = (3.8 + min(0.6, (swing_power - 1.0) * 0.15)) * sweet_mult
                    vertical_lift = 1.90 + max(0.0, p_norm[1] * 0.40)
                    ball.spin_x = float(np.clip(player_paddle.vy * 4.5 - blade_pitch * 4.0, -8.0, -1.8))
                elif is_attack_motion:
                    if player_paddle.vx > 0.20 or "FOREHAND" in stroke_name:
                        shot_type = "GANCHO FOREHAND"
                    elif player_paddle.vx < -0.20 or "BACKHAND" in stroke_name:
                        shot_type = "GANCHO BACKHAND"
                    else:
                        shot_type = "GANCHO DRIVE"
                    forward_speed = (4.0 + min(0.8, (swing_power - 1.0) * 0.20)) * sweet_mult
                    vertical_lift = 1.85 + p_norm[1] * 0.40
                    upward_vy = max(0.2, player_paddle.vy)
                    ball.spin_x = float(np.clip(upward_vy * 4.0 + 2.5 - blade_pitch * 3.0, 2.0, 8.5))
                else:
                    shot_type = "BLOQUEIO"
                    forward_speed = 3.8 * sweet_mult
                    vertical_lift = 1.80 + p_norm[1] * 0.35
                    ball.spin_x = 1.5

                if abs(hit_x) >= 0.40:
                    shot_type = f"{shot_type} (CANTO)"
                if sweet_tier == "SWEET_SPOT":
                    shot_type = f"★ {shot_type}"

                # Sidespin from wrist roll and lateral racket velocity
                wrist_spin = getattr(player_paddle, "roll", 0.0) * 8.0 if player_paddle.hand_detected else 0.0
                raw_sidespin = player_paddle.vx * 6.5 + wrist_spin
                if is_attack_motion:
                    if "FOREHAND" in stroke_name or player_paddle.vx > 0.25:
                        raw_sidespin = max(raw_sidespin, 5.0 + swing_power * 1.5)
                    elif "BACKHAND" in stroke_name or player_paddle.vx < -0.25:
                        raw_sidespin = min(raw_sidespin, -5.0 - swing_power * 1.5)
                ball.spin_y = float(np.clip(raw_sidespin, -14.0, 14.0))

                normal_bias_x = float(p_norm[0] * 0.50) if player_paddle.hand_detected else 0.0
                target_vx = float(np.clip(
                    0.15 * ball.vx + player_paddle.vx * 0.32 + normal_bias_x + timing_angle,
                    -0.85,
                    0.85,
                ))

                ball.x = hit_x
                ball.y = hit_y
                ball.vx = target_vx
                ball.vy = vertical_lift
                ball.vz = forward_speed
                # Position ball cleanly in front of paddle heading back towards Roberval
                ball.z = hit_z + self.ball_config.RADIUS + 0.02

                ball.last_hitter = "PLAYER"
                ball.bounce_count_current_hit = 0
                ball.bounced_player_side = False
                ball.bounced_ai_side = False
                ball.is_serve = False
                ball.squash = 0.42 if is_smash else 0.30

                events.append(PhysicsEvent(
                    event_type="HIT",
                    hitter="PLAYER",
                    position=(ball.x, ball.y, ball.z),
                    velocity=(ball.vx, ball.vy, ball.vz),
                    speed_kmh=ball.speed_kmh,
                    details=shot_type,
                ))

        # Roberval Paddle: Accurate continuous collision detection at AI strike plane
        is_crossing_ai = (prev_z <= ai_paddle.z and ball.z >= ai_paddle.z)
        is_near_ai_plane = abs(ball.z - ai_paddle.z) <= (self.ball_config.RADIUS + 0.05)

        if ball.vz > 0.0 and (is_crossing_ai or (is_near_ai_plane and ball.z >= 1.20)):
            if abs(ball.z - prev_z) > 1e-6:
                t_hit = float(np.clip((ai_paddle.z - prev_z) / (ball.z - prev_z), 0.0, 1.0))
                hit_x = float(prev_x + t_hit * (ball.x - prev_x))
                hit_y = float(prev_y + t_hit * (ball.y - prev_y))
            else:
                hit_x = float(ball.x)
                hit_y = float(ball.y)

            dx = hit_x - ai_paddle.x
            dy = hit_y - ai_paddle.y
            radial_dist = math.sqrt(dx * dx + dy * dy)

            vis_r = getattr(ai_paddle.config, "VISUAL_RADIUS", 0.076)
            ai_max_reach = vis_r * 1.80  # ~13.5cm reach for Roberval

            if radial_dist <= ai_max_reach:
                ball.x = hit_x
                ball.y = hit_y
                ball.vz = -3.8
                ball.vy = 1.85
                ball.vx = float(np.clip(0.18 * ball.vx + (hit_x - ai_paddle.x) * 1.2, -0.65, 0.65))
                # Position ball cleanly on the front rubber face of Roberval's racket
                ball.z = ai_paddle.z - self.ball_config.RADIUS - 0.015
                ball.spin_x = 1.0
                ball.spin_y = 0.0

                ball.last_hitter = "ROBERVAL"
                ball.bounce_count_current_hit = 0
                ball.bounced_player_side = False
                ball.bounced_ai_side = False
                ball.is_serve = False
                ball.squash = 0.30

                events.append(PhysicsEvent(
                    event_type="HIT",
                    hitter="ROBERVAL",
                    position=(ball.x, ball.y, ball.z),
                    velocity=(ball.vx, ball.vy, ball.vz),
                    speed_kmh=ball.speed_kmh,
                    details="RETURN",
                ))

        # 3. Check Net Collision (Decisive ITTF Net Resolution: Never gets stuck)
        if self.table.check_net_collision(prev_z, ball.z, ball.x, ball.y, ball.radius):
            # If ball clipped the very top tape (top 2cm of net), let it roll over
            if ball.y >= (self.table.net_height - 0.025):
                # Preserves forward direction, slight drag
                forward_sign = 1.0 if prev_z < ball.z else -1.0
                ball.vz = forward_sign * max(2.0, abs(ball.vz) * 0.75)
                ball.vy = max(0.5, ball.vy * 0.7)
                ball.z += 0.04 * forward_sign  # Nudge past net plane
                events.append(PhysicsEvent(
                    event_type="NET_CLIP",
                    hitter=ball.last_hitter,
                    position=(ball.x, ball.y, ball.z),
                    speed_kmh=ball.speed_kmh,
                    details="NET_CLIP",
                ))
            else:
                # Ball hit mesh below tape: Definite Net Fault!
                ball.in_play = False
                ball.vz = -ball.vz * 0.25  # Rebound gently off net mesh
                ball.vy = 0.40
                events.append(PhysicsEvent(
                    event_type="NET_FAULT",
                    hitter=ball.last_hitter,
                    position=(ball.x, ball.y, ball.z),
                    speed_kmh=ball.speed_kmh,
                    details="NET_FAULT",
                ))

        # 4. Table Surface Bounce Collision
        if ball.y <= ball.radius:
            # Check if contact is on the table surface
            if self.table.is_on_table(ball.x, ball.z):
                # Clamp above surface to prevent sinking
                ball.y = ball.radius
                # Restitution bounce
                ball.vy = -self.ball_config.RESTITUTION_TABLE * ball.vy
                ball.squash = 0.28

                # Minimum bounce threshold to prevent infinite tiny bounces
                if abs(ball.vy) < 0.25:
                    ball.vy = 0.0

                # Friction with table surface alters horizontal velocity
                ball.vx *= 0.95
                ball.vz *= 0.96

                # Spin reaction: Surface friction kicks velocity based on spin
                ball.vx += ball.spin_y * 0.025
                ball.vz += ball.spin_x * 0.025
                ball.spin_x *= 0.6
                ball.spin_y *= 0.6

                ball.bounce_count_current_hit += 1

                # Determine side of bounce
                side = "PLAYER" if self.table.is_player_side(ball.z) else "AI"
                if side == "PLAYER":
                    ball.bounced_player_side = True
                else:
                    ball.bounced_ai_side = True

                events.append(PhysicsEvent(
                    event_type="BOUNCE",
                    hitter=ball.last_hitter,
                    position=(ball.x, ball.y, ball.z),
                    velocity=(ball.vx, ball.vy, ball.vz),
                    speed_kmh=ball.speed_kmh,
                    details=side,
                ))

        # 5. Out of bounds check
        if ball.y < -0.40 or abs(ball.z) > 2.8 or abs(ball.x) > 1.8:
            ball.in_play = False
            events.append(PhysicsEvent(
                event_type="OUT",
                hitter=ball.last_hitter,
                position=(ball.x, ball.y, ball.z),
                speed_kmh=ball.speed_kmh,
                details="Ball out of play",
            ))

        return events

    def step_dead_ball(self, ball: Ball, dt: float):
        """Advance ball tumbling naturally under gravity after point is decided."""
        ball.vy -= self.ball_config.GRAVITY * dt
        ball.x += ball.vx * dt
        ball.y += ball.vy * dt
        ball.z += ball.vz * dt

        # Floor bounce (-0.76m is floor height below table surface)
        floor_y = -0.76 + ball.radius
        if ball.y <= floor_y:
            ball.y = floor_y
            ball.vy = -ball.vy * 0.40
            ball.vx *= 0.80
            ball.vz *= 0.80
            if abs(ball.vy) < 0.2:
                ball.vy = 0.0

        # Table surface bounce if on table
        if abs(ball.x) <= self.table.half_width and abs(ball.z) <= self.table.half_length:
            if ball.y <= ball.radius and ball.vy < 0:
                ball.y = ball.radius
                ball.vy = -ball.vy * 0.45
                ball.vx *= 0.85
                ball.vz *= 0.85

        # Update motion trail so the dead ball doesn't leave disconnected ghosts
        ball.trail.append((ball.x, ball.y, ball.z))
        if len(ball.trail) > 10:
            ball.trail.pop(0)
