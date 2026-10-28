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
            spin_vec = np.array([ball.spin_x, ball.spin_y, ball.spin_z], dtype=np.float64)
            omega_cross_v = np.cross(spin_vec, v)
            k_magnus = 0.00040 / self.ball_config.MASS
            accel += k_magnus * omega_cross_v

        # Integrate velocity and position (Velocity Verlet / Symplectic Euler)
        ball.vx += accel[0] * dt
        ball.vy += accel[1] * dt
        ball.vz += accel[2] * dt

        ball.x += ball.vx * dt
        ball.y += ball.vy * dt
        ball.z += ball.vz * dt

        # 2. Check Paddle Collisions (Calibrated Wii Sports Rebatidas)
        # Player Paddle: Hook Swing / Smash / Drive / Block Interception
        if ball.vz < 0.0 and ball.z <= -0.75:
            dx = ball.x - player_paddle.x
            dy = ball.y - player_paddle.y
            radial_dist = math.sqrt(dx * dx + dy * dy)
            z_dist = abs(ball.z - player_paddle.z)

            # Check if player performed a hook swing or smash
            swing_speed = player_paddle.speed
            is_hook = (swing_speed >= 0.90) or (abs(player_paddle.vx) >= 0.55)
            is_smash = (swing_speed >= 2.8 and player_paddle.vy < -0.3) or (player_paddle.vz > 4.0) or (swing_speed >= 3.8)

            # Assisted reach: 48cm for active hook swing, 40cm for block
            max_reach = 0.48 if is_hook else 0.40

            if radial_dist <= max_reach and z_dist <= 0.42:
                if is_smash:
                    shot_type = "GANCHO SMASH"
                    forward_speed = min(5.6, 4.6 + swing_speed * 0.20)
                    vertical_lift = 1.10
                    ball.spin_x = 12.0  # Heavy topspin
                    ball.spin_y = (ball.x - player_paddle.x) * 8.0
                elif is_hook:
                    if player_paddle.vx > 0.4:
                        shot_type = "GANCHO FOREHAND"
                        forward_speed = 3.7 + min(1.0, swing_speed * 0.18)
                        vertical_lift = 1.65
                        ball.spin_y = 6.0
                        ball.spin_x = 6.0
                    elif player_paddle.vx < -0.4:
                        shot_type = "GANCHO BACKHAND"
                        forward_speed = 3.6 + min(1.0, swing_speed * 0.18)
                        vertical_lift = 1.65
                        ball.spin_y = -6.0
                        ball.spin_x = 6.0
                    else:
                        shot_type = "GANCHO DRIVE"
                        forward_speed = 3.6 + min(1.0, swing_speed * 0.18)
                        vertical_lift = 1.65
                        ball.spin_x = 4.0
                else:
                    shot_type = "BLOQUEIO"
                    forward_speed = 3.3
                    vertical_lift = 1.60

                # Directional horizontal control: Racket placement + stroke momentum
                offset_x = ball.x - player_paddle.x
                target_vx = float(np.clip(
                    0.15 * ball.vx + offset_x * 1.5 + player_paddle.vx * 0.35,
                    -0.85,
                    0.85,
                ))

                ball.vx = target_vx
                ball.vy = vertical_lift
                ball.vz = forward_speed
                # Nudge ball slightly in front of paddle to prevent multiple triggers in consecutive sub-steps
                ball.z = max(ball.z, player_paddle.z + 0.06)

                ball.last_hitter = "PLAYER"
                ball.bounce_count_current_hit = 0
                ball.bounced_player_side = False
                ball.bounced_ai_side = False
                ball.is_serve = False

                events.append(PhysicsEvent(
                    event_type="HIT",
                    hitter="PLAYER",
                    position=(ball.x, ball.y, ball.z),
                    velocity=(ball.vx, ball.vy, ball.vz),
                    speed_kmh=ball.speed_kmh,
                    details=shot_type,
                ))

        # Roberval Paddle: Calibrated return arc that reaches player table and racket
        elif ball.vz > 0.0 and ball.z >= 0.75:
            dx = ball.x - ai_paddle.x
            dy = ball.y - ai_paddle.y
            radial_dist = math.sqrt(dx * dx + dy * dy)
            z_dist = abs(ball.z - ai_paddle.z)

            if radial_dist <= 0.45 and z_dist <= 0.40:
                # Calibrated return: clears net smoothly, lands comfortably on player table, reaches baseline
                ball.vz = -3.5
                ball.vy = 1.70
                ball.vx = float(np.clip(0.20 * ball.vx + (ball.x - ai_paddle.x) * 1.3, -0.65, 0.65))
                ball.z = min(ball.z, ai_paddle.z - 0.06)

                ball.last_hitter = "ROBERVAL"
                ball.bounce_count_current_hit = 0
                ball.bounced_player_side = False
                ball.bounced_ai_side = False
                ball.is_serve = False

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
                ball.vz = 0.0
                ball.vy = -0.5
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

                # Minimum bounce threshold to prevent infinite tiny bounces
                if abs(ball.vy) < 0.25:
                    ball.vy = 0.0

                # Friction with table surface alters horizontal velocity
                ball.vx *= 0.95
                ball.vz *= 0.96

                # Spin reaction
                ball.vz += ball.spin_x * 0.02
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
