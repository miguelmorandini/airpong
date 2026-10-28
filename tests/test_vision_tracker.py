"""Unit tests for vision tracker, gesture classifier, and camera sources."""

import numpy as np
import pytest

from src.vision.camera import SyntheticCameraSource
from src.vision.gestures import GestureClassifier
from src.vision.tracker import HandTracker


def test_synthetic_camera_source():
    cam = SyntheticCameraSource(width=320, height=240, fps=30)
    assert cam.is_opened is True

    ret, frame = cam.read()
    assert ret is True
    assert frame is not None
    assert frame.shape == (240, 320, 3)

    cam.release()
    assert cam.is_opened is False


def test_gesture_classifier():
    classifier = GestureClassifier(right_handed=True)

    # Idle velocity
    idle_v = np.array([0.1, 0.1, 0.1])
    accel = np.array([0.0, 0.0, 0.0])
    res_idle = classifier.classify(idle_v, accel)
    assert res_idle.stroke_type == "IDLE"
    assert res_idle.is_active_swing is False

    # Forehand swing (positive vx for right-handed)
    fh_v = np.array([3.5, 0.5, 0.5])
    res_fh = classifier.classify(fh_v, accel)
    assert "FOREHAND" in res_fh.stroke_type
    assert res_fh.is_active_swing is True
    assert res_fh.is_hook_swing is True

    # Backhand swing (negative vx for right-handed)
    bh_v = np.array([-3.5, 0.5, 0.5])
    res_bh = classifier.classify(bh_v, accel)
    assert "BACKHAND" in res_bh.stroke_type
    assert res_bh.is_active_swing is True
    assert res_bh.is_hook_swing is True

    # Power smash (high speed, downward vy)
    smash_v = np.array([1.0, -2.5, 7.0])
    res_smash = classifier.classify(smash_v, accel)
    assert "SMASH" in res_smash.stroke_type
    assert res_smash.is_active_swing is True
    assert res_smash.is_hook_swing is True


def test_hand_tracker_with_frame():
    tracker = HandTracker()
    # Test with synthetic frame
    cam = SyntheticCameraSource(width=640, height=480)
    ret, frame = cam.read()
    assert ret is True

    result = tracker.process_frame(frame)
    # Check bounded world coordinates
    assert -2.0 <= result.world_x <= 2.0
    assert -0.2 <= result.world_y <= 1.0
    assert -2.5 <= result.world_z <= 0.0
    assert result.annotated_frame is not None
    assert hasattr(result, "hook_power")
    assert hasattr(result, "is_hook_swing")
    assert isinstance(result.hook_power, (int, float))
    assert isinstance(result.is_hook_swing, bool)
    cam.release()
