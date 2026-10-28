"""Configuration and constants for AI Ping Pong."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TableConfig:
    """Official ITTF Table Tennis proportions (meters)."""
    # Origin (0,0,0) is at the center of the table top surface.
    LENGTH: float = 2.74          # Z axis: -1.37 to +1.37
    WIDTH: float = 1.525          # X axis: -0.7625 to +0.7625
    SURFACE_Y: float = 0.0        # Y axis: Table top surface is at 0
    NET_HEIGHT: float = 0.1525    # Height of net above surface
    NET_OVERHANG: float = 0.1525  # Net extends sideways beyond table width
    BORDER_WIDTH: float = 0.02    # White edge line width (meters)
    CENTER_LINE_WIDTH: float = 0.003  # White center dividing line

    @property
    def half_length(self) -> float:
        return self.LENGTH / 2.0

    @property
    def half_width(self) -> float:
        return self.WIDTH / 2.0


@dataclass(frozen=True)
class BallConfig:
    """Official ITTF Ball specifications."""
    RADIUS: float = 0.02          # 40mm diameter ball -> 20mm radius
    MASS: float = 0.0027          # 2.7 grams
    RESTITUTION_TABLE: float = 0.88    # Bounciness on table surface
    RESTITUTION_PADDLE: float = 0.92   # Bounciness on rubber paddle
    AIR_DENSITY: float = 1.204         # kg/m^3 (sea level, 20°C)
    DRAG_COEFFICIENT: float = 0.44     # Sphere aerodynamic drag
    GRAVITY: float = 9.81              # m/s^2 downward


@dataclass(frozen=True)
class PaddleConfig:
    """Paddle geometry and collision bounds calibrated for hand-to-table interception."""
    RADIUS: float = 0.28          # 28 cm blade radius for reliable block interception
    ASSISTED_RADIUS: float = 0.42 # Forgiving strike reach
    HANDLE_LENGTH: float = 0.10   # Handle length
    MAX_SWING_SPEED: float = 18.0 # Max registered swing speed in m/s
    PLAYER_Z_REST: float = -1.40  # Player baseline strike plane (exact table end)
    AI_Z_REST: float = 1.40       # AI opponent baseline plane


@dataclass(frozen=True)
class RulesConfig:
    """Official ITTF Scoring Rules."""
    POINTS_TO_WIN: int = 11
    MIN_POINT_DIFFERENCE: int = 2
    SERVES_PER_ROTATION: int = 2
    DEUCE_SCORE: int = 10


@dataclass(frozen=True)
class CameraConfig:
    """Camera and vision processing settings."""
    WIDTH: int = 640
    HEIGHT: int = 480
    FPS: int = 60
    SMOOTHING_ALPHA: float = 0.35  # EMA smoothing factor for hand tracking
    MIN_DETECTION_CONFIDENCE: float = 0.65
    MIN_TRACKING_CONFIDENCE: float = 0.65


@dataclass(frozen=True)
class DisplayConfig:
    """Rendering dimensions and Wii Sports palette."""
    CANVAS_WIDTH: int = 860
    CANVAS_HEIGHT: int = 580
    TABLE_COLOR_RGB: Tuple[int, int, int] = (16, 92, 168)      # Classic ITTF Blue
    TABLE_BORDER_RGB: Tuple[int, int, int] = (245, 245, 245)   # Crisp White
    NET_COLOR_RGB: Tuple[int, int, int] = (220, 220, 225)      # Semi-transparent mesh
    BALL_COLOR_RGB: Tuple[int, int, int] = (255, 165, 0)       # Vibrant 3-star Orange
    PADDLE_RED_RGB: Tuple[int, int, int] = (220, 40, 40)       # Red rubber
    PADDLE_BLACK_RGB: Tuple[int, int, int] = (30, 30, 30)      # Black rubber
    BG_GRADIENT_TOP: Tuple[int, int, int] = (180, 215, 245)    # Wii Sports light blue
    BG_GRADIENT_BOTTOM: Tuple[int, int, int] = (230, 240, 250)
