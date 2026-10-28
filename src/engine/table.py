"""Table tennis geometry and boundary checks according to ITTF specifications."""

from dataclasses import dataclass
from typing import Tuple
from src.config import TableConfig


@dataclass
class Table:
    """3D Table model and boundary detection."""
    config: TableConfig = TableConfig()

    @property
    def half_length(self) -> float:
        return self.config.half_length

    @property
    def half_width(self) -> float:
        return self.config.half_width

    @property
    def net_height(self) -> float:
        return self.config.NET_HEIGHT

    @property
    def net_overhang(self) -> float:
        return self.config.NET_OVERHANG

    def is_on_table(self, x: float, z: float, margin: float = 0.0) -> bool:
        """Check if (x, z) coordinates fall on the table surface."""
        return (
            -self.half_width - margin <= x <= self.half_width + margin
            and -self.half_length - margin <= z <= self.half_length + margin
        )

    def is_player_side(self, z: float) -> bool:
        """Player human occupies negative Z space (Z in [-1.37, 0])."""
        return -self.half_length <= z < 0.0

    def is_ai_side(self, z: float) -> bool:
        """AI opponent occupies positive Z space (Z in (0, 1.37])."""
        return 0.0 < z <= self.half_length

    def check_net_collision(
        self,
        z_prev: float,
        z_curr: float,
        x: float,
        y: float,
        ball_radius: float,
    ) -> bool:
        """Check if the ball trajectory crosses the net plane at Z=0 below net height."""
        # Check if ball crossed Z=0 in this frame
        if (z_prev < 0.0 <= z_curr) or (z_prev > 0.0 >= z_curr) or abs(z_curr) < 0.015:
            # Check within net width (including overhang)
            max_x = self.half_width + self.net_overhang
            if -max_x <= x <= max_x:
                # Check height from surface (0) to net top
                if y - ball_radius <= self.net_height:
                    return True
        return False

    def get_corners_3d(self) -> Tuple[Tuple[float, float, float], ...]:
        """Return the 4 corners of the table surface for 3D projection rendering."""
        w2 = self.half_width
        l2 = self.half_length
        y = self.config.SURFACE_Y
        return (
            (-w2, y, -l2),  # Player left corner
            (w2, y, -l2),   # Player right corner
            (w2, y, l2),    # AI right corner
            (-w2, y, l2),   # AI left corner
        )

    def get_net_posts_3d(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return 3D coordinates for top-left and top-right of the net."""
        overhang_w = self.half_width + self.net_overhang
        ny = self.config.SURFACE_Y + self.net_height
        return (
            (-overhang_w, ny, 0.0),
            (overhang_w, ny, 0.0),
        )
