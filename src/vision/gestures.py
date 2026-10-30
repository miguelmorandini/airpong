"""Gesture and stroke classifier for table tennis paddle swings."""

import os
import json
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


@dataclass
class GestureResult:
    stroke_type: str         # "GANCHO_FOREHAND", "GANCHO_BACKHAND", "GANCHO_SMASH", "DRIVE", "CHOP", "SERVE", "IDLE"
    swing_speed: float       # m/s
    swing_speed_kmh: float   # km/h
    is_active_swing: bool
    is_hook_swing: bool      # True when a dynamic hook motion (gancho) is detected
    hook_power: float        # Power multiplier (1.0x to 2.5x)
    confidence: float
    model_source: str = "HEURISTIC" # "ML_MODEL", "ARM_KINEMATICS", "HEURISTIC"
    elbow_angle: float = 0.0
    arm_power: float = 1.0
    kinetic_phase: str = "IDLE"     # "IDLE", "WINDUP", "SNAP", "FOLLOW_THROUGH"


class GestureClassifier:
    """Classifies physical hand and arm movements into table tennis strokes using

    either trained ML models or calibrated biomechanical kinematics rules.
    """

    def __init__(
        self,
        right_handed: bool = True,
        model_path: Optional[str] = None,
        use_ml_if_available: bool = False,
    ):
        self.right_handed = right_handed
        self.swing_threshold = 0.60    # m/s threshold for deliberate hook motion (forgiving natural motion)
        self.smash_threshold = 2.40    # m/s for power smash hook
        self.use_ml_if_available = use_ml_if_available

        # ML model references
        default_dir = os.path.join(os.path.dirname(__file__), "models")
        self.model_path = model_path or os.path.join(default_dir, "stroke_classifier.joblib")
        self.meta_path = os.path.join(default_dir, "stroke_model_meta.json")

        self.ml_model = None
        self.ml_features = []
        self.profile = None

        # Kinematic history for backswing/wind-up detection
        self._prev_speeds = []
        self._prev_vz = []

        if self.use_ml_if_available:
            self._try_load_ml_model()

    def apply_profile(self, profile):
        """Apply user calibration profile to customize gesture detection thresholds."""
        self.profile = profile
        if profile is not None:
            self.right_handed = (getattr(profile, "handedness", "RIGHT") == "RIGHT")
            self.swing_threshold = getattr(profile, "forehand_speed_threshold", 1.05)
            self.smash_threshold = getattr(profile, "smash_speed_threshold", 2.40)

    def _try_load_ml_model(self):
        """Attempt to load trained scikit-learn model."""
        if not JOBLIB_AVAILABLE:
            return
        if os.path.exists(self.model_path):
            try:
                self.ml_model = joblib.load(self.model_path)
                if os.path.exists(self.meta_path):
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.ml_features = meta.get("features", [])
            except Exception:
                self.ml_model = None

    def classify(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        is_fist: bool = False,
        hand_scale: float = 0.15,
        palm_normal: Optional[Tuple[float, float, float]] = None,
        arm_angle: Optional[float] = None,
        is_above_shoulder: bool = False,
        arm_power: float = 1.0,
        arm_angular_speed: float = 0.0,
        **kwargs,
    ) -> GestureResult:
        """Analyze hand & arm velocity, acceleration, and orientation to classify table tennis strokes."""
        vx, vy, vz = float(velocity[0]), float(velocity[1]), float(velocity[2])
        speed = float(np.linalg.norm(velocity))

        # Kinetic chain multiplier combines arm angle lever and angular whip
        k_power = arm_power
        if abs(arm_angular_speed) > 40.0:
            k_power = min(2.5, k_power + (abs(arm_angular_speed) - 40.0) * 0.008)

        effective_speed = speed * k_power
        effective_speed_kmh = effective_speed * 3.6
        speed_kmh = speed * 3.6
        current_elbow = arm_angle or 0.0

        # Maintain short history for kinetic phase analysis
        self._prev_speeds.append(speed)
        self._prev_vz.append(vz)
        if len(self._prev_speeds) > 10:
            self._prev_speeds.pop(0)
            self._prev_vz.pop(0)

        # Kinetic phase detection
        kinetic_phase = "IDLE"
        if len(self._prev_vz) >= 4 and vz < -0.3 and speed > 0.4:
            kinetic_phase = "WINDUP"
        elif speed >= self.swing_threshold and vz > 0.3:
            kinetic_phase = "SNAP"
        elif speed >= self.swing_threshold * 0.6:
            kinetic_phase = "FOLLOW_THROUGH"

        if speed < self.swing_threshold and effective_speed < self.swing_threshold:
            return GestureResult(
                stroke_type="IDLE",
                swing_speed=speed,
                swing_speed_kmh=speed_kmh,
                is_active_swing=False,
                is_hook_swing=False,
                hook_power=1.0,
                confidence=0.85,
                model_source="HEURISTIC",
                elbow_angle=current_elbow,
                arm_power=k_power,
                kinetic_phase=kinetic_phase,
            )

        hook_power = max(1.0, min(2.5, 1.0 + (effective_speed - self.swing_threshold) * 0.45))

        # 1. Unambiguous directional kinematic invariants:
        # High speed downward/forward snap or overhead arm position is Smash
        is_smash_arm = is_above_shoulder and (effective_speed >= self.smash_threshold * 0.65 or vy < -0.35)
        if (effective_speed >= self.smash_threshold and (vy < -0.6 or vz > 1.8)) or is_smash_arm:
            return GestureResult(
                stroke_type="GANCHO_SMASH",
                swing_speed=effective_speed,
                swing_speed_kmh=effective_speed_kmh,
                is_active_swing=True,
                is_hook_swing=True,
                hook_power=hook_power * 1.30 * k_power,
                confidence=0.96,
                model_source="ARM_KINEMATICS" if is_above_shoulder or arm_angle is not None else "HEURISTIC",
                elbow_angle=current_elbow,
                arm_power=k_power,
                kinetic_phase="SNAP",
            )

        # Clear lateral arm sweep is Forehand or Backhand
        lat_vx = vx if self.right_handed else -vx
        if lat_vx > 0.8:
            return GestureResult(
                stroke_type="GANCHO_FOREHAND",
                swing_speed=effective_speed,
                swing_speed_kmh=effective_speed_kmh,
                is_active_swing=True,
                is_hook_swing=True,
                hook_power=hook_power * k_power,
                confidence=0.93,
                model_source="ARM_KINEMATICS" if arm_angle is not None else "HEURISTIC",
                elbow_angle=current_elbow,
                arm_power=k_power,
                kinetic_phase="SNAP",
            )
        elif lat_vx < -0.8:
            return GestureResult(
                stroke_type="GANCHO_BACKHAND",
                swing_speed=effective_speed,
                swing_speed_kmh=effective_speed_kmh,
                is_active_swing=True,
                is_hook_swing=True,
                hook_power=hook_power * k_power,
                confidence=0.93,
                model_source="ARM_KINEMATICS" if arm_angle is not None else "HEURISTIC",
                elbow_angle=current_elbow,
                arm_power=k_power,
                kinetic_phase="SNAP",
            )

        # 2. For nuanced / complex gestures (CHOP, SERVE, DRIVE), consult ML model
        if self.ml_model is not None:
            try:
                nx, ny, nz = palm_normal or (0.0, 0.0, 1.0)
                feat_map = {
                    "vx": vx if self.right_handed else -vx,
                    "vy": vy,
                    "vz": vz,
                    "speed": speed,
                    "ax": float(acceleration[0]),
                    "ay": float(acceleration[1]),
                    "az": float(acceleration[2]),
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "palm_nx": nx,
                    "palm_ny": ny,
                    "palm_nz": nz,
                    "hand_scale": hand_scale,
                    "is_fist": int(is_fist),
                }
                feature_order = self.ml_features or list(feat_map.keys())
                vec = [float(feat_map.get(k, 0.0)) for k in feature_order]
                x_in = np.array([vec], dtype=np.float32)

                pred_stroke = str(self.ml_model.predict(x_in)[0])
                probs = self.ml_model.predict_proba(x_in)[0]
                idx = list(self.ml_model.classes_).index(pred_stroke)
                conf = float(probs[idx])

                if conf >= 0.50:
                    is_smash = "SMASH" in pred_stroke
                    return GestureResult(
                        stroke_type=pred_stroke,
                        swing_speed=effective_speed,
                        swing_speed_kmh=effective_speed_kmh,
                        is_active_swing=True,
                        is_hook_swing=True,
                        hook_power=hook_power * (1.25 if is_smash else 1.0) * k_power,
                        confidence=conf,
                        model_source="ML_MODEL",
                        elbow_angle=current_elbow,
                        arm_power=k_power,
                        kinetic_phase=kinetic_phase,
                    )
            except Exception:
                pass  # Graceful fallback to rule-based logic below

        # Rule-based fallback classification
        # Defensive Chop: downward velocity with open blade (pitch > 0.15)
        if vy < -0.45 and pitch > 0.12 and abs(lat_vx) < 0.8:
            return GestureResult(
                stroke_type="CHOP",
                swing_speed=effective_speed,
                swing_speed_kmh=effective_speed_kmh,
                is_active_swing=True,
                is_hook_swing=True,
                hook_power=hook_power * 0.95,
                confidence=0.88,
                model_source="ARM_KINEMATICS" if arm_angle is not None else "HEURISTIC",
                elbow_angle=current_elbow,
                arm_power=k_power,
                kinetic_phase="SNAP",
            )

        # Forehand Hook vs Backhand Hook based on calibrated lateral arm sweep
        fh_threshold = getattr(self.profile, "forehand_speed_threshold", 0.75) * 0.70 if self.profile else 0.75
        bh_threshold = getattr(self.profile, "backhand_speed_threshold", 0.75) * 0.70 if self.profile else 0.75

        if self.right_handed:
            if vx > fh_threshold:
                stroke = "GANCHO_FOREHAND"
            elif vx < -bh_threshold:
                stroke = "GANCHO_BACKHAND"
            else:
                stroke = "DRIVE"
        else:
            if vx < -fh_threshold:
                stroke = "GANCHO_FOREHAND"
            elif vx > bh_threshold:
                stroke = "GANCHO_BACKHAND"
            else:
                stroke = "DRIVE"

        return GestureResult(
            stroke_type=stroke,
            swing_speed=effective_speed,
            swing_speed_kmh=effective_speed_kmh,
            is_active_swing=True,
            is_hook_swing=True,
            hook_power=hook_power * k_power,
            confidence=0.90,
            model_source="ARM_KINEMATICS" if arm_angle is not None else "HEURISTIC",
            elbow_angle=current_elbow,
            arm_power=k_power,
            kinetic_phase=kinetic_phase,
        )
