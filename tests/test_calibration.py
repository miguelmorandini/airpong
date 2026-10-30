"""Unit tests for UserGestureProfile and GestureTrainer calibration flow."""

import os
import tempfile
import pytest
from src.vision.calibration import UserGestureProfile, GestureTrainer
from src.vision.gestures import GestureClassifier, GestureResult


def test_user_gesture_profile_defaults():
    profile = UserGestureProfile()
    assert profile.handedness == "RIGHT"
    assert profile.serve_speed_threshold == 0.85
    assert profile.forehand_speed_threshold == 1.05
    assert profile.backhand_speed_threshold == 1.05
    assert profile.smash_speed_threshold == 2.40
    assert not profile.is_calibrated


def test_user_gesture_profile_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_calib.json")
        profile = UserGestureProfile(
            handedness="LEFT",
            serve_speed_threshold=1.25,
            forehand_speed_threshold=1.55,
            backhand_speed_threshold=1.40,
            smash_speed_threshold=2.80,
            is_calibrated=True,
        )
        profile.save(test_path)

        loaded = UserGestureProfile.load(test_path)
        assert loaded.handedness == "LEFT"
        assert loaded.serve_speed_threshold == 1.25
        assert loaded.forehand_speed_threshold == 1.55
        assert loaded.is_calibrated is True


def test_gesture_trainer_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_trainer_calib.json")
        profile = UserGestureProfile()
        profile._save_path = test_path
        trainer = GestureTrainer(profile=profile)

        # Start trainer
        trainer.start()
        assert trainer.is_active is True
        assert trainer.current_step[0] == "REST"

        # Advance through REST
        trainer.step_start_time -= 3.0  # simulate 3s passed
        trainer.update(speed=0.1, vx=0.0, vy=0.0, is_fist=False, dt=0.016)
        assert trainer.current_step[0] == "SERVE"

        # Simulate 2 serve swings
        trainer.update(speed=1.5, vx=0.1, vy=0.6, is_fist=False, dt=0.016)
        trainer.last_swing_time -= 1.0  # simulate cooldown passed
        trainer.update(speed=1.7, vx=0.1, vy=0.7, is_fist=False, dt=0.016)
        assert trainer.current_step[0] == "FOREHAND"

        # Skip remaining steps
        trainer.skip_step()  # skip FOREHAND -> BACKHAND
        assert trainer.current_step[0] == "BACKHAND"
        trainer.skip_step()  # skip BACKHAND -> SMASH
        assert trainer.current_step[0] == "SMASH"
        trainer.skip_step()  # skip SMASH -> FINALIZE

        assert trainer.is_active is False
        assert profile.is_calibrated is True


def test_gesture_classifier_apply_profile():
    classifier = GestureClassifier()
    custom_profile = UserGestureProfile(
        serve_speed_threshold=0.60,
        forehand_speed_threshold=0.80,
        backhand_speed_threshold=0.80,
        smash_speed_threshold=1.50,
        is_calibrated=True,
    )
    classifier.apply_profile(custom_profile)
    assert classifier.profile == custom_profile

    # Test stroke classification with arm kinematics
    # High speed downward stroke above shoulder -> SMASH
    res = classifier.classify(
        velocity=(0.2, -1.6, 0.0),
        acceleration=(0.0, 0.0, 0.0),
        arm_angle=130.0,
        is_above_shoulder=True,
    )
    assert isinstance(res, GestureResult)
    assert res.stroke_type == "GANCHO_SMASH"
