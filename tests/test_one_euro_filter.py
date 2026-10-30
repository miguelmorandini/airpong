"""Unit tests for the One Euro (1€) adaptive low-pass filter."""

import numpy as np
import pytest

from src.vision.one_euro_filter import OneEuroFilter, OneEuroFilter3D


def test_one_euro_filter_steady_state():
    """Verify filter smooths out micro-jitter when input is steady with slight noise."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.01)

    np.random.seed(42)
    clean_val = 0.5
    filtered_vals = []

    t = 0.0
    dt = 1.0 / 60.0
    for _ in range(60):
        noisy_val = clean_val + np.random.normal(0, 0.05)
        out = f.filter(noisy_val, timestamp=t, dt=dt)
        filtered_vals.append(out)
        t += dt

    # Steady state output should be very close to the true value (jitter reduced)
    steady_mean = np.mean(filtered_vals[30:])
    steady_std = np.std(filtered_vals[30:])
    assert steady_mean == pytest.approx(clean_val, abs=0.03)
    # Output variance should be significantly smaller than input noise variance (0.05)
    assert steady_std < 0.02


def test_one_euro_filter_step_response():
    """Verify filter quickly tracks sudden changes with dynamic cutoff (low lag)."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)

    # Initial state at 0.0
    t = 0.0
    dt = 1.0 / 60.0
    for _ in range(10):
        f.filter(0.0, timestamp=t, dt=dt)
        t += dt

    # Rapid jump to 1.0
    outputs = []
    for _ in range(15):
        out = f.filter(1.0, timestamp=t, dt=dt)
        outputs.append(out)
        t += dt

    # Filter should rapidly converge towards 1.0 within 15 frames (~250ms)
    assert outputs[-1] > 0.85


def test_one_euro_filter_3d():
    """Verify 3D vector filter works across all axes."""
    f3d = OneEuroFilter3D(min_cutoff=1.2, beta=0.02)
    x, y, z = f3d.filter(0.1, 0.2, -1.4, dt=1.0 / 60.0)
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert isinstance(z, float)
    assert x == pytest.approx(0.1, abs=0.01)
    assert y == pytest.approx(0.2, abs=0.01)
    assert z == pytest.approx(-1.4, abs=0.01)
