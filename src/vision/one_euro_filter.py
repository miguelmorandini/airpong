"""One Euro (1€) Filter for real-time jitter reduction and zero-lag motion tracking.

Reference: Casiez, G., Roussel, N., & Vogel, D. (2012).
1€ filter: a simple speed-based low-pass filter for noisy input in interactive systems.
ACM CHI 2012.
"""

import math
from typing import Optional, Tuple


class LowPassFilter:
    """Standard first-order exponential low-pass filter."""

    def __init__(self, alpha: float = 1.0, init_val: Optional[float] = None):
        self.alpha = float(alpha)
        self.s = init_val
        self.has_init = init_val is not None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = float(alpha)
        if not self.has_init:
            self.s = float(value)
            self.has_init = True
            return self.s
        self.s = self.alpha * float(value) + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self):
        self.has_init = False
        self.s = None


class OneEuroFilter:
    """
    Adaptive low-pass filter with dynamic cutoff frequency.
    Eliminates micro-jitter when the hand is still, and reduces latency to zero during rapid swings.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ):
        """
        Args:
            min_cutoff: Minimum cutoff frequency in Hz. Lower = smoother at rest.
            beta: Speed coefficient. Higher = faster response / lower lag during rapid motion.
            d_cutoff: Cutoff frequency for derivative filter in Hz.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: Optional[float] = None, dt: Optional[float] = None) -> float:
        """Filter noisy input value x and return smoothed output."""
        if dt is None:
            if timestamp is None or self.last_time is None:
                dt = 1.0 / 60.0
            else:
                dt = max(1e-4, timestamp - self.last_time)
        if timestamp is not None:
            self.last_time = timestamp

        # Estimate derivative of the signal
        if self.x_filter.has_init:
            dx = (x - self.x_filter.s) / dt
        else:
            dx = 0.0

        # Filter the derivative
        alpha_d = self._alpha(self.d_cutoff, dt)
        edx = self.dx_filter.filter(dx, alpha_d)

        # Compute dynamic cutoff frequency based on speed
        cutoff = self.min_cutoff + self.beta * abs(edx)

        # Filter the value with the adaptive cutoff
        alpha = self._alpha(cutoff, dt)
        return self.x_filter.filter(x, alpha)

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class OneEuroFilter3D:
    """Tri-axis One Euro Filter for 3D coordinates (X, Y, Z)."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.015, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fz = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def filter(self, x: float, y: float, z: float, timestamp: Optional[float] = None, dt: Optional[float] = None) -> Tuple[float, float, float]:
        return (
            self.fx.filter(x, timestamp, dt),
            self.fy.filter(y, timestamp, dt),
            self.fz.filter(z, timestamp, dt),
        )

    def reset(self):
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()
