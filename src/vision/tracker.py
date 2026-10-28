"""MediaPipe Hand Tracker with coordinate projection, EMA smoothing and paddle mapping."""

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


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"


@dataclass
class HandTrackingResult:
    detected: bool
    # 3D coordinates mapped to table tennis space (meters)
    world_x: float = 0.0
    world_y: float = 0.20
    world_z: float = -1.65
    # Velocity in 3D (m/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    speed_kmh: float = 0.0
    # Normalized 2D coordinates [0..1]
    norm_x: float = 0.5
    norm_y: float = 0.5
    hand_scale: float = 0.15
    gesture: GestureResult = None
    annotated_frame: Optional[np.ndarray] = None


class HandTracker:
    """Tracks player hand landmarks in 3D space to control virtual paddle."""

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
        self._ensure_model_file()

        # Initialize MediaPipe Tasks HandLandmarker
        self.detector: Optional[vision.HandLandmarker] = None
        self._init_detector()

        # State tracking for smoothing & velocity calculation
        self.prev_x = 0.0
        self.prev_y = 0.20
        self.prev_z = -1.65
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_vz = 0.0
        self.last_timestamp = time.time()
        self.alpha = self.camera_config.SMOOTHING_ALPHA

    def _ensure_model_file(self):
        """Download hand_landmarker.task model if not present."""
        if not os.path.exists(self.model_path):
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
            except Exception:
                # If download fails, will fallback gracefully
                pass

    def _init_detector(self):
        """Create MediaPipe HandLandmarker detector instance."""
        if not os.path.exists(self.model_path):
            self.detector = None
            return

        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=self.camera_config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.camera_config.MIN_TRACKING_CONFIDENCE,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception:
            self.detector = None

    def process_frame(self, frame: np.ndarray) -> HandTrackingResult:
        """Process BGR frame and return 3D paddle coordinates."""
        now = time.time()
        dt = max(1e-4, now - self.last_timestamp)
        self.last_timestamp = now

        h, w = frame.shape[:2]
        annotated = frame.copy()

        # If MediaPipe detector is not available, return fallback center position
        if self.detector is None:
            return self._fallback_result(annotated, dt)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            detection_result = self.detector.detect(mp_image)
        except Exception:
            return self._fallback_result(annotated, dt)

        if not detection_result.hand_landmarks:
            return self._fallback_result(annotated, dt)

        # Extract landmarks for the primary hand
        landmarks = detection_result.hand_landmarks[0]

        # Landmark 0: Wrist, 9: Middle Finger MCP, 5: Index MCP, 17: Pinky MCP
        wrist = landmarks[0]
        middle_mcp = landmarks[9]

        # Palm center: weighted average of wrist and middle knuckle
        palm_nx = 0.4 * wrist.x + 0.6 * middle_mcp.x
        palm_ny = 0.4 * wrist.y + 0.6 * middle_mcp.y

        # Depth estimate from hand scale in camera view
        # Distance between wrist and middle knuckle
        hand_scale = math.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)
        hand_scale = max(0.06, min(0.35, hand_scale))

        # Map normalized coordinates to 3D Ping Pong Table coordinates:
        # X: 0 (left) to 1 (right) -> -0.95m to +0.95m
        target_x = (palm_nx - 0.5) * 1.90
        # Y: 0 (top) to 1 (bottom) -> inverted: higher hand in frame = higher paddle (0.05m to 0.70m)
        target_y = (1.0 - palm_ny) * 0.75 - 0.05
        # Z: larger scale (closer to camera) = closer to player table side (-1.45m to -1.80m)
        target_z = -1.80 + (hand_scale - 0.08) * 1.2
        target_z = max(-1.85, min(-1.38, target_z))

        # EMA Smoothing
        smoothed_x = self.alpha * target_x + (1.0 - self.alpha) * self.prev_x
        smoothed_y = self.alpha * target_y + (1.0 - self.alpha) * self.prev_y
        smoothed_z = self.alpha * target_z + (1.0 - self.alpha) * self.prev_z

        # Velocity
        vx = (smoothed_x - self.prev_x) / dt
        vy = (smoothed_y - self.prev_y) / dt
        vz = (smoothed_z - self.prev_z) / dt

        # Smooth velocity
        smooth_vx = 0.5 * vx + 0.5 * self.prev_vx
        smooth_vy = 0.5 * vy + 0.5 * self.prev_vy
        smooth_vz = 0.5 * vz + 0.5 * self.prev_vz

        self.prev_x, self.prev_y, self.prev_z = smoothed_x, smoothed_y, smoothed_z
        self.prev_vx, self.prev_vy, self.prev_vz = smooth_vx, smooth_vy, smooth_vz

        velocity_vec = np.array([smooth_vx, smooth_vy, smooth_vz], dtype=np.float64)
        accel_vec = velocity_vec / dt

        gesture = self.gesture_classifier.classify(velocity_vec, accel_vec)

        # Draw hand landmarks & paddle overlay on annotated frame
        self._draw_landmarks(annotated, landmarks, w, h)
        self._draw_paddle_hud(annotated, palm_nx, palm_ny, gesture, w, h)

        return HandTrackingResult(
            detected=True,
            world_x=float(smoothed_x),
            world_y=float(smoothed_y),
            world_z=float(smoothed_z),
            vx=float(smooth_vx),
            vy=float(smooth_vy),
            vz=float(smooth_vz),
            speed_kmh=gesture.swing_speed_kmh,
            norm_x=float(palm_nx),
            norm_y=float(palm_ny),
            hand_scale=float(hand_scale),
            gesture=gesture,
            annotated_frame=annotated,
        )

    def _fallback_result(self, annotated: np.ndarray, dt: float) -> HandTrackingResult:
        """Fallback when no hand is in frame."""
        # Gently drift towards resting stance
        rest_x, rest_y, rest_z = 0.0, 0.20, -1.65
        decay = 0.1
        self.prev_x = (1.0 - decay) * self.prev_x + decay * rest_x
        self.prev_y = (1.0 - decay) * self.prev_y + decay * rest_y
        self.prev_z = (1.0 - decay) * self.prev_z + decay * rest_z
        self.prev_vx *= 0.5
        self.prev_vy *= 0.5
        self.prev_vz *= 0.5

        cv2.putText(
            annotated,
            "POSITION HAND IN FRONT OF CAMERA",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

        idle_gesture = GestureResult("IDLE", 0.0, 0.0, False, 1.0)
        return HandTrackingResult(
            detected=False,
            world_x=self.prev_x,
            world_y=self.prev_y,
            world_z=self.prev_z,
            vx=0.0,
            vy=0.0,
            vz=0.0,
            speed_kmh=0.0,
            norm_x=0.5,
            norm_y=0.5,
            hand_scale=0.15,
            gesture=idle_gesture,
            annotated_frame=annotated,
        )

    def _draw_landmarks(self, frame: np.ndarray, landmarks, w: int, h: int):
        """Draw hand skeletal connections and joints."""
        # 21 landmarks
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        # Connections
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

    def _draw_paddle_hud(self, frame: np.ndarray, nx: float, ny: float, gesture: GestureResult, w: int, h: int):
        """Draw virtual paddle blade and speed indicator."""
        px, py = int(nx * w), int(ny * h)
        # Paddle blade circle
        cv2.circle(frame, (px, py), 30, (30, 30, 220), -1)      # Red rubber
        cv2.circle(frame, (px, py), 30, (255, 255, 255), 2)     # White rim
        cv2.line(frame, (px, py + 25), (px, py + 55), (100, 70, 50), 7)  # Wood handle

        # Display stroke & speed
        hud_text = f"{gesture.stroke_type}: {gesture.swing_speed_kmh:.1f} km/h"
        color = (0, 255, 100) if gesture.is_active_swing else (200, 200, 200)
        cv2.putText(frame, hud_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
