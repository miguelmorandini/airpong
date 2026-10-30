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
    assert hasattr(result, "roll")
    assert hasattr(result, "pitch")
    assert hasattr(result, "yaw")
    assert hasattr(result, "normal")
    assert len(result.normal) == 3
    assert isinstance(result.hook_power, (int, float))
    assert isinstance(result.is_hook_swing, bool)
    assert hasattr(result, "arm_power")
    assert hasattr(result, "elbow_angle")
    assert hasattr(result, "kinetic_phase")
    assert hasattr(result, "is_grip")
    assert hasattr(result, "grip_type")
    assert hasattr(result, "blade_center_norm")
    assert hasattr(result, "playing_arm")
    assert hasattr(result, "both_shoulders_detected")
    assert isinstance(result.is_grip, bool)
    assert isinstance(result.grip_type, str)
    assert isinstance(result.playing_arm, str)
    cam.release()


def test_handedness_and_dual_shoulder_tracking():
    """Verify handedness cycling and adaptive dual-shoulder arm tracking."""
    tracker = HandTracker()
    assert tracker.handedness_mode == "AUTO"

    # Cycle handedness: AUTO -> RIGHT -> LEFT -> AUTO
    mode1 = tracker.toggle_handedness()
    assert mode1 == "RIGHT"
    assert tracker.gesture_classifier.right_handed is True

    mode2 = tracker.toggle_handedness()
    assert mode2 == "LEFT"
    assert tracker.gesture_classifier.right_handed is False

    mode3 = tracker.toggle_handedness()
    assert mode3 == "AUTO"


def test_gesture_classifier_arm_kinetics():
    classifier = GestureClassifier(right_handed=True)
    v = np.array([2.5, 0.4, 0.4])
    accel = np.array([5.0, 1.0, 1.0])

    # Test with arm power multiplier & angular whip
    res = classifier.classify(
        v, accel, arm_angle=125.0, arm_power=1.35, arm_angular_speed=65.0
    )
    assert res.is_active_swing is True
    assert res.arm_power > 1.30
    assert "FOREHAND" in res.stroke_type

    # Test Smash with overhead arm position
    v_smash = np.array([0.2, -0.6, 2.0])
    res_smash = classifier.classify(
        v_smash, accel, is_above_shoulder=True, arm_power=1.40
    )
    assert "SMASH" in res_smash.stroke_type
    assert res_smash.hook_power > 1.5


def test_paddle_arm_kinetics_update():
    from src.engine.paddle import Paddle
    paddle = Paddle(is_player=True)
    paddle.update_position(
        target_x=0.10,
        target_y=0.25,
        target_z=-1.35,
        dt=0.016,
        arm_power=1.42,
        elbow_angle=118.0,
        arm_detected=True,
        kinetic_phase="SNAP",
    )
    assert paddle.arm_detected is True
    assert paddle.arm_power == 1.42
    assert paddle.elbow_angle == 118.0
    assert paddle.kinetic_phase == "SNAP"


def test_hand_tracker_detected_flow():
    """Verify that process_frame executes all branches without NameError when hands and pose are detected."""
    class MockLM:
        def __init__(self, x=0.5, y=0.5, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class MockHandRes:
        def __init__(self, lms):
            self.hand_landmarks = [lms]

    class MockPoseRes:
        def __init__(self, lms):
            self.pose_landmarks = [lms]

    tracker = HandTracker()

    # 21 hand landmarks
    hand_lms = [MockLM(0.65, 0.45, 0.0) for _ in range(21)]
    # Set wrist (0), index MCP (5), middle MCP (9), pinky MCP (17)
    hand_lms[0] = MockLM(0.65, 0.55, 0.0)
    hand_lms[5] = MockLM(0.62, 0.42, 0.0)
    hand_lms[9] = MockLM(0.65, 0.40, 0.0)
    hand_lms[17] = MockLM(0.68, 0.44, 0.0)

    # 33 pose landmarks
    pose_lms = [MockLM(0.5, 0.5, 0.0) for _ in range(33)]
    pose_lms[11] = MockLM(0.65, 0.30, 0.0)  # Right shoulder on mirrored screen
    pose_lms[12] = MockLM(0.35, 0.30, 0.0)  # Left shoulder on mirrored screen
    pose_lms[13] = MockLM(0.70, 0.42, 0.0)  # Right elbow
    pose_lms[14] = MockLM(0.30, 0.42, 0.0)  # Left elbow
    pose_lms[15] = MockLM(0.65, 0.55, 0.0)  # Right wrist
    pose_lms[16] = MockLM(0.35, 0.55, 0.0)  # Left wrist

    class MockDetector:
        def detect(self, mp_img):
            return MockHandRes(hand_lms)

    class MockPoseDetector:
        def detect(self, mp_img):
            return MockPoseRes(pose_lms)

    tracker.detector = MockDetector()
    tracker.pose_detector = MockPoseDetector()

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = tracker.process_frame(dummy_frame)

    assert res.detected is True
    assert res.both_shoulders_detected is True
    assert res.playing_arm == "DIREITO"
    assert res.arm_detected is True
    assert isinstance(res.is_backhand_face, bool)
    assert res.annotated_frame is not None
