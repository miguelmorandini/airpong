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
            ball.x = 0.20
            ball.y = 0.22
            ball.z = -1.50
            ball.vx = -0.3
            ball.vy = 1.4
            ball.vz = 4.2
        else:
            ball.x = -0.20
            ball.y = 0.22
            ball.z = 1.50
            ball.vx = 0.3
            ball.vy = 1.4
            ball.vz = -4.2

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

        # 2. Check Paddle Collisions
        # Player Paddle collision check (only if ball is heading towards player side or in zone)
        if ball.vz < 0.0 and ball.z < -1.1:
            hit, dist, z_dist = player_paddle.check_ball_collision(ball.position, ball.radius)
            if hit:
                new_v, shot_type = player_paddle.compute_rebound_velocity(
                    ball.velocity, ball.position, self.ball_config.RESTITUTION_PADDLE
                )
                ball.vx, ball.vy, ball.vz = new_v
                ball.last_hitter = "PLAYER"
                ball.bounce_count_current_hit = 0
                ball.bounced_player_side = False
                ball.bounced_ai_side = False
                ball.is_serve = False
                # Add topspin if swing had upward vy
                ball.spin_x = player_paddle.vy * 3.5

                events.append(PhysicsEvent(
                    event_type="HIT",
                    hitter="PLAYER",
                    position=(ball.x, ball.y, ball.z),
                    velocity=(ball.vx, ball.vy, ball.vz),
                    speed_kmh=ball.speed_kmh,
                    details=shot_type,
                ))

        # AI Paddle collision check (only if ball is heading towards AI side or in zone)
        elif ball.vz > 0.0 and ball.z > 1.1:
            hit, dist, z_dist = ai_paddle.check_ball_collision(ball.position, ball.radius)
            if hit:
                new_v, shot_type = ai_paddle.compute_rebound_velocity(
                    ball.velocity, ball.position, self.ball_config.RESTITUTION_PADDLE
                )
                ball.vx, ball.vy, ball.vz = new_v
                ball.last_hitter = "AI"
                ball.bounce_count_current_hit = 0
                ball.bounced_player_side = False
                ball.bounced_ai_side = False
                ball.is_serve = False
                ball.spin_x = ai_paddle.vy * 3.5

                events.append(PhysicsEvent(
                    event_type="HIT",
                    hitter="AI",
                    position=(ball.x, ball.y, ball.z),
                    velocity=(ball.vx, ball.vy, ball.vz),
                    speed_kmh=ball.speed_kmh,
                    details=shot_type,
                ))

        # 3. Check Net Collision
        if self.table.check_net_collision(prev_z, ball.z, ball.x, ball.y, ball.radius):
            # Ball hit net tape or net cord
            # Dampen velocity and deflect
            ball.vz = -0.35 * ball.vz
            ball.vy = max(0.4, ball.vy * 0.5)
            ball.vx *= 0.6
            events.append(PhysicsEvent(
                event_type="NET",
                hitter=ball.last_hitter,
                position=(ball.x, ball.y, ball.z),
                speed_kmh=ball.speed_kmh,
                details="Ball collided with net",
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
