"""MediaPipe Hand & Arm Kinetic Tracker with coordinate projection, One Euro filtering and paddle mapping."""

import math
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import CameraConfig, TableConfig
from src.vision.gestures import GestureClassifier, GestureResult
from src.vision.one_euro_filter import OneEuroFilter, OneEuroFilter3D


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_FILENAME = "pose_landmarker_lite.task"


@dataclass
class HandTrackingResult:
    detected: bool
    # 3D coordinates mapped to table tennis space (meters)
    world_x: float = 0.0
    world_y: float = 0.20
    world_z: float = -1.40
    # Velocity in 3D (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    speed_kmh: float = 0.0
    # True 3D wrist/palm orientation angles (radians)
    roll: float = 0.0    # Sideways wrist tilt (angle of knuckle line)
    pitch: float = 0.0   # Forward/backward blade face tilt
    yaw: float = 0.0     # Lateral blade aiming direction
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0) # Palm normal vector
    # Normalized 2D coordinates [0..1]
    norm_x: float = 0.5
    norm_y: float = 0.5
    hand_scale: float = 0.15
    is_fist: bool = False
    # Grip stance modeling (Holding a physical or virtual paddle)
    is_grip: bool = True
    grip_type: str = "CLASSICO"       # "CLASSICO", "CANETA", "CLASSINETA", "FIST", "OPEN"
    is_backhand_face: bool = False     # True if backhand (black) face is angled forward
    blade_center_norm: Tuple[float, float] = (0.5, 0.5)
    # Arm kinematics & Upper Body
    arm_detected: bool = False
    both_shoulders_detected: bool = False
    playing_hand: str = "direita"
    playing_arm: str = "DIREITO"       # "DIREITO" or "ESQUERDO"
    elbow_angle: float = 0.0
    arm_power: float = 1.0
    arm_angular_speed: float = 0.0
    kinetic_phase: str = "IDLE"
    shoulder_pos: Optional[Tuple[float, float, float]] = None
    shoulder_left_pos: Optional[Tuple[float, float, float]] = None
    shoulder_right_pos: Optional[Tuple[float, float, float]] = None
    elbow_pos: Optional[Tuple[float, float, float]] = None
    gesture: Optional[GestureResult] = None
    annotated_frame: Optional[np.ndarray] = None

    @property
    def is_hook_swing(self) -> bool:
        if self.gesture:
            return getattr(self.gesture, "is_hook_swing", False)
        return False

    @property
    def hook_power(self) -> float:
        if self.gesture and hasattr(self.gesture, "swing_speed"):
            if self.gesture.swing_speed > 0:
                return float(self.gesture.swing_speed)
        return float(math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz))


class HandTracker:
    """Tracks player hand landmarks & arm kinetic chain in 3D space to control virtual paddle."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        camera_config: Optional[CameraConfig] = None,
        table_config: Optional[TableConfig] = None,
    ):
        self.camera_config = camera_config or CameraConfig()
        self.table_config = table_config or TableConfig()
        self.gesture_classifier = GestureClassifier(right_handed=True)

        self.model_path = model_path or os.path.join(os.getcwd(), MODEL_FILENAME)
        self.pose_model_path = os.path.join(os.getcwd(), POSE_MODEL_FILENAME)
        self._ensure_model_file()

        # Initialize MediaPipe Tasks HandLandmarker & PoseLandmarker
        self.detector: Optional[vision.HandLandmarker] = None
        self.pose_detector: Optional[vision.PoseLandmarker] = None
        self._init_detector()

        # State tracking for smoothing & velocity calculation
        self.prev_x = 0.0
        self.prev_y = 0.20
        self.prev_z = -1.40
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_vz = 0.0
        self.last_timestamp = time.time()
        self.alpha = self.camera_config.SMOOTHING_ALPHA

        # One Euro Filter for adaptive zero-lag smoothing & jitter elimination
        self.pos_filter = OneEuroFilter3D(min_cutoff=1.2, beta=0.018, d_cutoff=1.0)
        self.roll_filter = OneEuroFilter(min_cutoff=1.5, beta=0.02, d_cutoff=1.0)
        self.pitch_filter = OneEuroFilter(min_cutoff=1.5, beta=0.02, d_cutoff=1.0)
        self.yaw_filter = OneEuroFilter(min_cutoff=1.5, beta=0.02, d_cutoff=1.0)
        self.auto_depth = True
        self.prev_roll = 0.0
        self.prev_pitch = 0.0
        self.prev_yaw = 0.0

        # Arm tracking state
        self.prev_elbow_angle = 120.0
        self.arm_filter = OneEuroFilter(min_cutoff=1.2, beta=0.015, d_cutoff=1.0)
        self.handedness_mode = "AUTO"  # "AUTO", "RIGHT", "LEFT"

    def toggle_handedness(self) -> str:
        """Cycle handedness mode: AUTO -> RIGHT -> LEFT -> AUTO."""
        modes = ["AUTO", "RIGHT", "LEFT"]
        idx = (modes.index(self.handedness_mode) + 1) % len(modes)
        self.handedness_mode = modes[idx]
        if self.handedness_mode == "RIGHT":
            self.gesture_classifier.right_handed = True
        elif self.handedness_mode == "LEFT":
            self.gesture_classifier.right_handed = False
        return self.handedness_mode

    def _ensure_model_file(self):
        """Download hand_landmarker.task and pose_landmarker_lite.task models if not present."""
        if not os.path.exists(self.model_path):
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
            except Exception:
                pass
        if not os.path.exists(self.pose_model_path):
            try:
                urllib.request.urlretrieve(POSE_MODEL_URL, self.pose_model_path)
            except Exception:
                pass

    def _init_detector(self):
        """Create MediaPipe HandLandmarker and PoseLandmarker detector instances."""
        if os.path.exists(self.model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=self.model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,  # Multi-hand support for dominant hand selection
                    min_hand_detection_confidence=self.camera_config.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=self.camera_config.MIN_TRACKING_CONFIDENCE,
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
            except Exception:
                self.detector = None
        else:
            self.detector = None

        if os.path.exists(self.pose_model_path):
            try:
                pose_base_opts = python.BaseOptions(model_asset_path=self.pose_model_path)
                pose_opts = vision.PoseLandmarkerOptions(
                    base_options=pose_base_opts,
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                self.pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)
            except Exception:
                self.pose_detector = None
        else:
            self.pose_detector = None

    def _classify_grip(self, landmarks, wrist, middle_mcp) -> Tuple[bool, str]:
        """Classify table tennis paddle grip: Clássico (Shakehand), Caneta (Penhold), or Closed Fist."""
        d_wrist = lambda idx: math.hypot(landmarks[idx].x - wrist.x, landmarks[idx].y - wrist.y)

        # Folded state of each finger: tip curled closer to wrist than PIP joint
        f_index_curled = d_wrist(8) < d_wrist(6) * 1.12
        f_middle_curled = d_wrist(12) < d_wrist(10) * 1.15
        f_ring_curled = d_wrist(16) < d_wrist(14) * 1.15
        f_pinky_curled = d_wrist(20) < d_wrist(18) * 1.15

        curled_count = sum([f_index_curled, f_middle_curled, f_ring_curled, f_pinky_curled])

        if curled_count >= 3:
            # Shakehand grip with index along blade base
            if not f_index_curled and f_middle_curled and f_ring_curled:
                return True, "CLASSICO"
            return True, "FIST"
        elif f_middle_curled and f_ring_curled and f_pinky_curled and not f_index_curled:
            # Classic Shakehand (forefinger extended along back)
            return True, "CLASSICO"
        elif f_ring_curled and f_pinky_curled and not f_middle_curled:
            # Japanese / Chinese Penhold grip (thumb & index pinch front)
            return True, "CANETA"
        elif curled_count >= 2:
            # Semi-gripped hand
            return True, "CLASSICO"
        else:
            return False, "OPEN"

    def process_frame(self, frame: np.ndarray) -> HandTrackingResult:
        """Process BGR frame and return 3D paddle coordinates with hand-arm fusion."""
        now = time.time()
        dt = max(1e-4, now - self.last_timestamp)
        self.last_timestamp = now

        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 1. Hand Detection
        detection_result = None
        if self.detector is not None:
            try:
                detection_result = self.detector.detect(mp_image)
            except Exception:
                detection_result = None

        has_hands = detection_result is not None and bool(detection_result.hand_landmarks)

        # 2. Pose Tracking (Dual Shoulder & Kinetic Arm Chain)
        arm_detected = False
        both_shoulders_detected = False
        elbow_angle = 120.0
        arm_angular_speed = 0.0
        is_above_shoulder = False
        arm_power = 1.0
        s_pos = None
        e_pos = None
        w_pose_lm = None
        e_pose_lm = None
        playing_arm_name = "DIREITO"
        shoulder_left_pos = None
        shoulder_right_pos = None

        if self.pose_detector is not None:
            try:
                pose_res = self.pose_detector.detect(mp_image)
                if pose_res.pose_landmarks:
                    p_lms = pose_res.pose_landmarks[0]
                    # In mirrored camera view (cv2.flip(frame, 1)):
                    # Landmark 11 (MediaPipe Left Shoulder) is on screen-right (User's physical Right Shoulder)
                    # Landmark 12 (MediaPipe Right Shoulder) is on screen-left (User's physical Left Shoulder)
                    s_right = p_lms[11]  # Physical Right Shoulder (Screen Right)
                    s_left = p_lms[12]   # Physical Left Shoulder (Screen Left)
                    e_right = p_lms[13]  # Physical Right Elbow
                    e_left = p_lms[14]   # Physical Left Elbow
                    w_right = p_lms[15]  # Physical Right Wrist
                    w_left = p_lms[16]   # Physical Left Wrist

                    both_shoulders_detected = True
                    shoulder_left_pos = (float(s_left.x), float(s_left.y), getattr(s_left, "z", 0.0))
                    shoulder_right_pos = (float(s_right.x), float(s_right.y), getattr(s_right, "z", 0.0))

                    # 2.1 Always draw athletic clavicle bar connecting BOTH shoulders!
                    slx_px, sly_px = int(s_left.x * w), int(s_left.y * h)
                    srx_px, sry_px = int(s_right.x * w), int(s_right.y * h)
                    cv2.line(annotated, (slx_px, sly_px), (srx_px, sry_px), (70, 95, 125), 2)
                    cv2.circle(annotated, (slx_px, sly_px), 5, (0, 200, 255), -1)  # Left shoulder dot
                    cv2.circle(annotated, (srx_px, sry_px), 5, (0, 200, 255), -1)  # Right shoulder dot

                    # 2.2 Determine active playing arm:
                    if self.handedness_mode == "RIGHT":
                        use_right = True
                    elif self.handedness_mode == "LEFT":
                        use_right = False
                    else:
                        # AUTO mode: match arm wrist closest to active hand wrist
                        if has_hands:
                            hw = detection_result.hand_landmarks[0][0]
                            dist_right = (hw.x - w_right.x)**2 + (hw.y - w_right.y)**2
                            dist_left = (hw.x - w_left.x)**2 + (hw.y - w_left.y)**2
                            use_right = (dist_right <= dist_left)
                        else:
                            use_right = getattr(self.gesture_classifier, "right_handed", True)

                    if use_right:
                        s_lm, e_lm, w_lm = s_right, e_right, w_right
                        other_s, other_e, other_w = s_left, e_left, w_left
                        playing_arm_name = "DIREITO"
                        self.gesture_classifier.right_handed = True
                    else:
                        s_lm, e_lm, w_lm = s_left, e_left, w_left
                        other_s, other_e, other_w = s_right, e_right, w_right
                        playing_arm_name = "ESQUERDO"
                        self.gesture_classifier.right_handed = False

                    w_pose_lm = w_lm
                    e_pose_lm = e_lm

                    # 2.3 Draw subtle outline for non-playing arm
                    ox_s, oy_s = int(other_s.x * w), int(other_s.y * h)
                    ox_e, oy_e = int(other_e.x * w), int(other_e.y * h)
                    ox_w, oy_w = int(other_w.x * w), int(other_w.y * h)
                    cv2.line(annotated, (ox_s, oy_s), (ox_e, oy_e), (50, 68, 88), 1)
                    cv2.line(annotated, (ox_e, oy_e), (ox_w, oy_w), (50, 68, 88), 1)
                    cv2.circle(annotated, (ox_e, oy_e), 4, (70, 92, 118), -1)

                    # 2.4 Compute elbow angle for playing arm: shoulder -> elbow -> wrist
                    v1 = np.array([s_lm.x - e_lm.x, s_lm.y - e_lm.y], dtype=np.float64)
                    v2 = np.array([w_lm.x - e_lm.x, w_lm.y - e_lm.y], dtype=np.float64)
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 1e-4 and norm2 > 1e-4:
                        cos_a = np.dot(v1, v2) / (norm1 * norm2)
                        raw_elbow_angle = float(math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0))))
                        elbow_angle = self.arm_filter.filter(raw_elbow_angle, timestamp=now, dt=dt)
                        arm_angular_speed = (elbow_angle - self.prev_elbow_angle) / dt
                        self.prev_elbow_angle = elbow_angle
                        arm_detected = True

                    is_above_shoulder = (w_lm.y < s_lm.y - 0.05)
                    s_pos = (float(s_lm.x), float(s_lm.y), getattr(s_lm, "z", 0.0))
                    e_pos = (float(e_lm.x), float(e_lm.y), getattr(e_lm, "z", 0.0))

                    # Kinetic multiplier based on biomechanical elbow leverage
                    if 75.0 <= elbow_angle <= 170.0:
                        arm_power = 1.0 + 0.38 * (1.0 - abs(elbow_angle - 128.0) / 50.0)

                    # Dynamic angular snap bonus (kinetic whip)
                    if abs(arm_angular_speed) > 50.0:
                        arm_power += min(0.60, (abs(arm_angular_speed) - 50.0) * 0.005)

                    # 2.5 Draw active playing arm with glowing kinetic colors
                    sx_px, sy_px = int(s_lm.x * w), int(s_lm.y * h)
                    ex_px, ey_px = int(e_lm.x * w), int(e_lm.y * h)
                    wx_px, wy_px = int(w_lm.x * w), int(w_lm.y * h)

                    if arm_power > 1.35 or abs(arm_angular_speed) > 60.0:
                        arm_col = (0, 140, 255)  # Orange/Gold
                    elif arm_power > 1.15:
                        arm_col = (0, 230, 255)  # Bright Yellow
                    else:
                        arm_col = (255, 200, 0)  # Cyan/Blue

                    cv2.line(annotated, (sx_px, sy_px), (ex_px, ey_px), arm_col, 3)
                    cv2.line(annotated, (ex_px, ey_px), (wx_px, wy_px), arm_col, 3)
                    cv2.circle(annotated, (sx_px, sy_px), 5, (0, 140, 255), -1)
                    cv2.circle(annotated, (ex_px, ey_px), 6, (0, 255, 255), -1)
                    cv2.circle(annotated, (wx_px, wy_px), 5, (0, 255, 120), -1)

                    cv2.putText(annotated, f"{int(elbow_angle)}deg", (ex_px + 8, ey_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, arm_col, 1)
                    cv2.putText(annotated, f"BRACO {playing_arm_name}", (sx_px - 25, sy_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 255), 1)
            except Exception:
                pass

        if not has_hands:
            # If HandLandmarker missed due to fast swing motion blur, but Pose is available,
            # synthesize palm from wrist + forearm vector to guarantee zero tracking dropouts!
            if arm_detected and w_pose_lm is not None and e_pose_lm is not None:
                return self._fusion_pose_result(
                    annotated, w_pose_lm, e_pose_lm, elbow_angle, is_above_shoulder,
                    arm_power, arm_angular_speed, s_pos, e_pos, w, h, dt, now,
                    playing_arm=playing_arm_name,
                    both_shoulders=both_shoulders_detected,
                    shoulder_left_pos=shoulder_left_pos,
                    shoulder_right_pos=shoulder_right_pos,
                )
            return self._fallback_result(
                annotated, dt,
                arm_detected=arm_detected,
                elbow_angle=elbow_angle,
                both_shoulders=both_shoulders_detected,
                playing_arm=playing_arm_name,
                shoulder_left_pos=shoulder_left_pos,
                shoulder_right_pos=shoulder_right_pos,
            )

        # Multi-hand selection: pick playing hand matching active playing arm wrist
        selected_landmarks = detection_result.hand_landmarks[0]
        if len(detection_result.hand_landmarks) > 1 and w_pose_lm is not None:
            best_dist = 999.0
            for lms in detection_result.hand_landmarks:
                d = (lms[0].x - w_pose_lm.x)**2 + (lms[0].y - w_pose_lm.y)**2
                if d < best_dist:
                    best_dist = d
                    selected_landmarks = lms

        landmarks = selected_landmarks
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        # Grip classification: Shakehand, Penhold, or Closed Fist around racket handle
        is_grip, grip_type = self._classify_grip(landmarks, wrist, middle_mcp)
        is_fist = (grip_type == "FIST")

        # Palm / Grip Anchor Position: weighted average of wrist and knuckles
        palm_nx = 0.35 * wrist.x + 0.65 * middle_mcp.x
        palm_ny = 0.35 * wrist.y + 0.65 * middle_mcp.y

        # Depth estimate from hand scale in camera view
        hand_scale = math.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)
        hand_scale = max(0.06, min(0.35, hand_scale))

        # Calibrated 1:1 Mapping between Camera Hand Space and Table Tennis 3D Space:
        # Natural comfortable arm box: X in [0.20, 0.80] (span 0.60), Y in [0.18, 0.82] (span 0.64)
        calib_x = np.clip((palm_nx - 0.20) / 0.60, 0.0, 1.0)
        calib_y = np.clip((palm_ny - 0.18) / 0.64, 0.0, 1.0)

        # X: Wide reach covering full table (-0.76m to +0.76m) and table corners comfortably (-0.95m to +0.95m)
        target_x = (calib_x - 0.5) * 1.90

        # Y: Table surface (0.04m) up to upper strike limit (0.55m)
        target_y = 0.04 + (1.0 - calib_y) * 0.51

        # True 3D wrist/palm orientation kinematics
        ux = middle_mcp.x - wrist.x
        uy = middle_mcp.y - wrist.y
        uz = getattr(middle_mcp, "z", 0.0) - getattr(wrist, "z", 0.0)

        vx_palm = landmarks[17].x - landmarks[5].x
        vy_palm = landmarks[17].y - landmarks[5].y
        vz_palm = getattr(landmarks[17], "z", 0.0) - getattr(landmarks[5], "z", 0.0)

        # Palm normal: cross product u x v
        nx = uy * vz_palm - uz * vy_palm
        ny = uz * vx_palm - ux * vz_palm
        nz = ux * vy_palm - uy * vx_palm
        n_norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if n_norm > 1e-5:
            nx /= n_norm
            ny /= n_norm
            nz /= n_norm
        else:
            nx, ny, nz = 0.0, 0.0, 1.0

        # Roll: angle across knuckle line (Index to Pinky)
        raw_roll = math.atan2(landmarks[17].y - landmarks[5].y, landmarks[17].x - landmarks[5].x)
        # Pitch: forward/backward blade face tilt
        raw_pitch = math.atan2(uz, math.sqrt(ux * ux + uy * uy + 1e-6))
        # Yaw: lateral paddle face aiming direction
        raw_yaw = math.atan2(ux, -uy)

        # Dynamic Z-depth: based on arm reach / scale (stepping in vs back)
        if self.auto_depth:
            norm_s = float(np.clip((hand_scale - 0.08) / 0.14, 0.0, 1.0))
            target_z = -1.55 + norm_s * 0.37
        else:
            target_z = -1.40

        # One Euro Filter adaptive smoothing: zero-jitter at rest & instant response in swing
        smoothed_x, smoothed_y, smoothed_z = self.pos_filter.filter(
            target_x, target_y, target_z, timestamp=now, dt=dt
        )
        smoothed_roll = self.roll_filter.filter(raw_roll, timestamp=now, dt=dt)
        smoothed_pitch = self.pitch_filter.filter(raw_pitch, timestamp=now, dt=dt)
        smoothed_yaw = self.yaw_filter.filter(raw_yaw, timestamp=now, dt=dt)

        # Velocity with deadband & teleport protection
        if not hasattr(self, "_has_prev_pos") or not self._has_prev_pos:
            self.prev_x, self.prev_y, self.prev_z = smoothed_x, smoothed_y, smoothed_z
            self.prev_vx, self.prev_vy, self.prev_vz = 0.0, 0.0, 0.0
            self._has_prev_pos = True
            vx, vy, vz = 0.0, 0.0, 0.0
        else:
            safe_dt = max(0.008, dt)
            vx = (smoothed_x - self.prev_x) / safe_dt
            vy = (smoothed_y - self.prev_y) / safe_dt
            vz = (smoothed_z - self.prev_z) / safe_dt

        # Deadband: sensor jitter below 0.18 m/s (~0.6 km/h) is pure camera resting noise
        inst_spd = math.sqrt(vx * vx + vy * vy + vz * vz)
        if inst_spd < 0.18:
            vx, vy, vz = 0.0, 0.0, 0.0
        elif inst_spd > 7.0:
            scale = 7.0 / inst_spd
            vx *= scale
            vy *= scale
            vz *= scale

        smooth_vx = 0.40 * vx + 0.60 * self.prev_vx
        smooth_vy = 0.40 * vy + 0.60 * self.prev_vy
        smooth_vz = 0.40 * vz + 0.60 * self.prev_vz

        if abs(smooth_vx) < 0.06: smooth_vx = 0.0
        if abs(smooth_vy) < 0.06: smooth_vy = 0.0
        if abs(smooth_vz) < 0.06: smooth_vz = 0.0

        self.prev_x, self.prev_y, self.prev_z = smoothed_x, smoothed_y, smoothed_z
        self.prev_vx, self.prev_vy, self.prev_vz = smooth_vx, smooth_vy, smooth_vz
        self.prev_roll, self.prev_pitch, self.prev_yaw = smoothed_roll, smoothed_pitch, smoothed_yaw

        velocity_vec = np.array([smooth_vx, smooth_vy, smooth_vz], dtype=np.float64)
        accel_vec = velocity_vec / dt

        gesture = self.gesture_classifier.classify(
            velocity_vec,
            accel_vec,
            roll=float(smoothed_roll),
            pitch=float(smoothed_pitch),
            yaw=float(smoothed_yaw),
            is_fist=is_fist,
            hand_scale=float(hand_scale),
            palm_normal=(float(nx), float(ny), float(nz)),
            arm_angle=elbow_angle if arm_detected else None,
            is_above_shoulder=is_above_shoulder,
            arm_power=arm_power,
            arm_angular_speed=arm_angular_speed,
        )

        # Determine if player is showing backhand rubber face (black) or forehand (red)
        is_right = getattr(self.gesture_classifier, "right_handed", True)
        is_backhand = (nz < -0.15) if is_right else (nz > 0.15)

        # Virtual blade center projects outward from knuckles along forearm direction
        u_len = math.hypot(ux, uy) or 1.0
        blade_cx = palm_nx + (ux / u_len) * 0.08
        blade_cy = palm_ny + (uy / u_len) * 0.08

        # Draw hand landmarks & paddle HUD overlay on annotated frame
        self._draw_landmarks(annotated, landmarks, w, h)
        self._draw_paddle_hud(
            annotated, wrist, middle_mcp, palm_nx, palm_ny,
            gesture, is_grip, grip_type, is_backhand,
            w, h, elbow_angle, arm_power, smoothed_roll
        )

        return HandTrackingResult(
            detected=True,
            world_x=float(smoothed_x),
            world_y=float(smoothed_y),
            world_z=float(smoothed_z),
            vx=float(smooth_vx),
            vy=float(smooth_vy),
            vz=float(smooth_vz),
            speed_kmh=gesture.swing_speed_kmh,
            roll=float(smoothed_roll),
            pitch=float(smoothed_pitch),
            yaw=float(smoothed_yaw),
            normal=(float(nx), float(ny), float(nz)),
            norm_x=float(palm_nx),
            norm_y=float(palm_ny),
            hand_scale=float(hand_scale),
            is_fist=is_fist,
            is_grip=is_grip,
            grip_type=grip_type,
            is_backhand_face=is_backhand,
            blade_center_norm=(float(blade_cx), float(blade_cy)),
            arm_detected=arm_detected,
            both_shoulders_detected=both_shoulders_detected,
            playing_hand=playing_arm_name.lower(),
            playing_arm=playing_arm_name,
            elbow_angle=elbow_angle,
            arm_power=arm_power,
            arm_angular_speed=arm_angular_speed,
            kinetic_phase=gesture.kinetic_phase,
            shoulder_pos=s_pos,
            shoulder_left_pos=shoulder_left_pos,
            shoulder_right_pos=shoulder_right_pos,
            elbow_pos=e_pos,
            gesture=gesture,
            annotated_frame=annotated,
        )

    def _fusion_pose_result(
        self,
        annotated: np.ndarray,
        wrist_lm,
        elbow_lm,
        elbow_angle: float,
        is_above_shoulder: bool,
        arm_power: float,
        arm_angular_speed: float,
        s_pos,
        e_pos,
        w: int,
        h: int,
        dt: float,
        now: float,
        playing_arm: str = "DIREITO",
        both_shoulders: bool = False,
        shoulder_left_pos: Optional[Tuple[float, float, float]] = None,
        shoulder_right_pos: Optional[Tuple[float, float, float]] = None,
    ) -> HandTrackingResult:
        """Synthesize hand position from forearm vector when hand detector is in motion blur."""
        dir_x = wrist_lm.x - elbow_lm.x
        dir_y = wrist_lm.y - elbow_lm.y
        dist = math.hypot(dir_x, dir_y) or 1.0
        dir_x /= dist
        dir_y /= dist

        palm_nx = wrist_lm.x + dir_x * 0.08
        palm_ny = wrist_lm.y + dir_y * 0.08

        calib_x = np.clip((palm_nx - 0.20) / 0.60, 0.0, 1.0)
        calib_y = np.clip((palm_ny - 0.18) / 0.64, 0.0, 1.0)

        target_x = (calib_x - 0.5) * 1.90
        target_y = 0.04 + (1.0 - calib_y) * 0.51
        target_z = -1.40

        smoothed_x, smoothed_y, smoothed_z = self.pos_filter.filter(
            target_x, target_y, target_z, timestamp=now, dt=dt
        )

        vx = (smoothed_x - self.prev_x) / dt
        vy = (smoothed_y - self.prev_y) / dt
        vz = (smoothed_z - self.prev_z) / dt
        smooth_vx = 0.6 * vx + 0.4 * self.prev_vx
        smooth_vy = 0.6 * vy + 0.4 * self.prev_vy
        smooth_vz = 0.6 * vz + 0.4 * self.prev_vz

        self.prev_x, self.prev_y, self.prev_z = smoothed_x, smoothed_y, smoothed_z
        self.prev_vx, self.prev_vy, self.prev_vz = smooth_vx, smooth_vy, smooth_vz

        velocity_vec = np.array([smooth_vx, smooth_vy, smooth_vz], dtype=np.float64)
        accel_vec = velocity_vec / dt

        gesture = self.gesture_classifier.classify(
            velocity_vec,
            accel_vec,
            roll=self.prev_roll,
            pitch=self.prev_pitch,
            yaw=self.prev_yaw,
            is_fist=True,
            hand_scale=0.15,
            arm_angle=elbow_angle,
            is_above_shoulder=is_above_shoulder,
            arm_power=arm_power,
            arm_angular_speed=arm_angular_speed,
        )

        self._draw_paddle_hud(
            annotated, wrist_lm, None, palm_nx, palm_ny,
            gesture, True, "CLASSICO", False,
            w, h, elbow_angle, arm_power, self.prev_roll
        )

        return HandTrackingResult(
            detected=True,
            world_x=float(smoothed_x),
            world_y=float(smoothed_y),
            world_z=float(smoothed_z),
            vx=float(smooth_vx),
            vy=float(smooth_vy),
            vz=float(smooth_vz),
            speed_kmh=gesture.swing_speed_kmh,
            roll=self.prev_roll,
            pitch=self.prev_pitch,
            yaw=self.prev_yaw,
            normal=(0.0, 0.0, 1.0),
            norm_x=float(palm_nx),
            norm_y=float(palm_ny),
            hand_scale=0.15,
            is_fist=True,
            is_grip=True,
            grip_type="CLASSICO",
            arm_detected=True,
            both_shoulders_detected=both_shoulders,
            playing_hand=playing_arm.lower(),
            playing_arm=playing_arm,
            elbow_angle=elbow_angle,
            arm_power=arm_power,
            arm_angular_speed=arm_angular_speed,
            kinetic_phase=gesture.kinetic_phase,
            shoulder_pos=s_pos,
            shoulder_left_pos=shoulder_left_pos,
            shoulder_right_pos=shoulder_right_pos,
            elbow_pos=e_pos,
            gesture=gesture,
            annotated_frame=annotated,
        )

    def _fallback_result(
        self,
        annotated: np.ndarray,
        dt: float,
        arm_detected: bool = False,
        elbow_angle: float = 120.0,
        both_shoulders: bool = False,
        playing_arm: str = "DIREITO",
        shoulder_left_pos: Optional[Tuple[float, float, float]] = None,
        shoulder_right_pos: Optional[Tuple[float, float, float]] = None,
    ) -> HandTrackingResult:
        """Fallback when no hand is in frame."""
        rest_x, rest_y, rest_z = 0.0, 0.22, -1.40
        decay = 0.1
        self.prev_x = (1.0 - decay) * self.prev_x + decay * rest_x
        self.prev_y = (1.0 - decay) * self.prev_y + decay * rest_y
        self.prev_z = (1.0 - decay) * self.prev_z + decay * rest_z
        self.prev_vx *= 0.5
        self.prev_vy *= 0.5
        self.prev_vz *= 0.5

        cv2.putText(
            annotated,
            "SEGURE A RAQUETE (MAO FECHADA/EMPUNHADURA) EM FRENTE A CAMERA",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 215, 255),
            2,
        )

        idle_gesture = GestureResult(
            stroke_type="IDLE",
            swing_speed=0.0,
            swing_speed_kmh=0.0,
            is_active_swing=False,
            is_hook_swing=False,
            hook_power=1.0,
            confidence=1.0,
        )
        self._has_prev_pos = False
        self.prev_vx *= 0.4
        self.prev_vy *= 0.4
        self.prev_vz *= 0.4
        if abs(self.prev_vx) < 0.05: self.prev_vx = 0.0
        if abs(self.prev_vy) < 0.05: self.prev_vy = 0.0
        if abs(self.prev_vz) < 0.05: self.prev_vz = 0.0

        return HandTrackingResult(
            detected=False,
            world_x=self.prev_x,
            world_y=self.prev_y,
            world_z=self.prev_z,
            vx=self.prev_vx,
            vy=self.prev_vy,
            vz=self.prev_vz,
            arm_detected=arm_detected,
            both_shoulders_detected=both_shoulders,
            playing_hand=playing_arm.lower(),
            playing_arm=playing_arm,
            elbow_angle=elbow_angle,
            shoulder_left_pos=shoulder_left_pos,
            shoulder_right_pos=shoulder_right_pos,
            gesture=idle_gesture,
            annotated_frame=annotated,
        )

    def _draw_landmarks(self, frame: np.ndarray, landmarks, w: int, h: int):
        """Draw hand skeletal connections and joints with glowing cyber aesthetics."""
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17),                               # Palm base
        ]
        for p1, p2 in connections:
            cv2.line(frame, coords[p1], coords[p2], (0, 220, 255), 2)
        for pt in coords:
            cv2.circle(frame, pt, 4, (255, 100, 0), -1)

    def _draw_paddle_hud(
        self,
        frame: np.ndarray,
        wrist_lm,
        middle_mcp,
        palm_nx: float,
        palm_ny: float,
        gesture: GestureResult,
        is_grip: bool,
        grip_type: str,
        is_backhand: bool,
        w: int,
        h: int,
        elbow_angle: float = 0.0,
        arm_power: float = 1.0,
        roll: float = 0.0,
    ):
        """Draw authentic table tennis paddle attached to the player's physical grip."""
        # 1. Calibration Box (Camera Hand Space -> Table Tennis Space)
        bx1, by1 = int(0.15 * w), int(0.18 * h)
        bx2, by2 = int(0.85 * w), int(0.82 * h)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 240, 120), 1)

        # Labels
        cv2.putText(frame, "AREA DE CONTROLE DA MESA", (bx1 + 10, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 120), 1)
        cv2.putText(frame, "ESQ", (bx1 + 5, by2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(frame, "CENTRO", (int(w / 2) - 25, by2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(frame, "DIR", (bx2 - 35, by2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # 2. Render Virtual Racket attached directly into the player's gripped hand
        wx, wy = int(wrist_lm.x * w), int(wrist_lm.y * h)
        if middle_mcp is not None:
            mx, my = int(middle_mcp.x * w), int(middle_mcp.y * h)
            vec_x, vec_y = mx - wx, my - wy
            v_len = math.hypot(vec_x, vec_y) or 1.0
            vec_x /= v_len
            vec_y /= v_len
        else:
            vec_x, vec_y = 0.0, -1.0

        # Racket blade center: projected outward past knuckles along grip axis
        blade_cx = int(wx + vec_x * 55)
        blade_cy = int(wy + vec_y * 55)

        # Handle angle in degrees
        angle_deg = math.degrees(math.atan2(vec_y, vec_x)) + 90.0

        # Draw wooden handle gripped inside the player's fist
        handle_start = (int(wx - vec_x * 15), int(wy - vec_y * 15))
        handle_end = (int(wx + vec_x * 25), int(wy + vec_y * 25))
        cv2.line(frame, handle_start, handle_end, (90, 60, 40), 10)     # Dark wood
        cv2.line(frame, handle_start, handle_end, (170, 120, 80), 6)    # Handle grip core
        cv2.line(frame, (blade_cx - int(vec_x * 10), blade_cy - int(vec_y * 10)), handle_end, (130, 95, 65), 8)

        # Draw Racket Blade (Rubber face + Protective rim)
        rubber_color = (25, 25, 25) if is_backhand else (20, 20, 215)  # Black on Backhand, Red on Forehand
        rubber_label = "BLACK (BH)" if is_backhand else "RED (FH)"

        # Draw blade outer rim
        cv2.ellipse(frame, (blade_cx, blade_cy), (32, 38), angle_deg, 0, 360, (235, 235, 235), 3)
        # Draw blade rubber face
        cv2.ellipse(frame, (blade_cx, blade_cy), (30, 36), angle_deg, 0, 360, rubber_color, -1)
        # Inner glossy reflection highlight
        cv2.ellipse(frame, (blade_cx - 5, blade_cy - 6), (12, 16), angle_deg, 0, 360, (255, 255, 255), 1)

        # Grip stance badge right on hand
        cv2.putText(
            frame,
            f"{grip_type}",
            (blade_cx - 24, blade_cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
        )

        # 3. Kinetic Power Meter (Top Right)
        meter_x = w - 170
        meter_y = 20
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + 150, meter_y + 14), (35, 40, 50), -1)
        fill_w = int(np.clip((arm_power - 1.0) / 1.5, 0.0, 1.0) * 150)
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + fill_w, meter_y + 14), (0, 215, 255), -1)
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + 150, meter_y + 14), (200, 200, 200), 1)
        cv2.putText(frame, f"KINETIC: {arm_power:.2f}x ({rubber_label})", (meter_x, meter_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1)

        # 4. Status and Stroke Badge (Bottom Left)
        if not gesture.is_active_swing:
            status_text = f"EMPUNHADURA {grip_type}: PRONTO PARA REBATER ({gesture.kinetic_phase})"
            color = (0, 240, 160)
        else:
            status_text = f"{gesture.stroke_type}: {gesture.swing_speed_kmh:.1f} km/h [{gesture.model_source}]"
            color = (0, 255, 100) if "SMASH" not in gesture.stroke_type else (0, 140, 255)

        cv2.putText(frame, status_text, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
