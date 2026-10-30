"""MediaPipe Hand Kinematics and Feature Extraction for Table Tennis Videos."""

import os
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import CameraConfig, TableConfig


@dataclass
class FrameKinematics:
    frame_index: int
    timestamp_sec: float
    detected: bool
    world_x: float
    world_y: float
    world_z: float
    vx: float
    vy: float
    vz: float
    speed: float
    speed_kmh: float
    ax: float
    ay: float
    az: float
    roll: float
    pitch: float
    yaw: float
    palm_nx: float
    palm_ny: float
    palm_nz: float
    hand_scale: float
    is_fist: bool
    stroke_label: str
    video_source: str


class HandKinematicsExtractor:
    """Extracts 21 3D landmarks and biomechanical stroke kinematics from table tennis video clips."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        camera_config: Optional[CameraConfig] = None,
        table_config: Optional[TableConfig] = None,
    ):
        self.camera_config = camera_config or CameraConfig()
        self.table_config = table_config or TableConfig()
        self.model_path = model_path or os.path.join(os.getcwd(), "hand_landmarker.task")

        self.detector: Optional[vision.HandLandmarker] = None
        self._init_detector()

    def _init_detector(self):
        """Initialize MediaPipe Hand Landmarker detector."""
        if not os.path.exists(self.model_path):
            self.detector = None
            return

        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[HandKinematicsExtractor] Falha ao iniciar MediaPipe detector: {e}")
            self.detector = None

    def process_video(
        self,
        video_path: str,
        stroke_label: str = "UNKNOWN",
        sample_step: int = 1,
    ) -> pd.DataFrame:
        """
        Extract frame-by-frame kinematics from a video file.
        Returns a structured pandas DataFrame ready for ML training.
        """
        if not os.path.exists(video_path):
            print(f"[HandKinematicsExtractor] Vídeo não encontrado: {video_path}")
            return pd.DataFrame()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[HandKinematicsExtractor] Não foi possível abrir o vídeo: {video_path}")
            return pd.DataFrame()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt = 1.0 / fps if fps > 0 else 0.033

        records: List[Dict[str, Any]] = []

        # Previous frame state for kinematics derivative
        prev_x, prev_y, prev_z = 0.0, 0.20, -1.40
        prev_vx, prev_vy, prev_vz = 0.0, 0.0, 0.0
        frame_idx = 0

        # Optical flow fallback trackers in case hand model misses stylized frames
        prev_gray: Optional[np.ndarray] = None

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if sample_step > 1 and (frame_idx % sample_step != 0):
                continue

            h, w = frame.shape[:2]
            ts = frame_idx * dt

            detected = False
            world_x, world_y, world_z = prev_x, prev_y, prev_z
            roll, pitch, yaw = 0.0, 0.0, 0.0
            palm_nx, palm_ny, palm_nz = 0.0, 0.0, 1.0
            hand_scale = 0.15
            is_fist = False

            # 1. Attempt MediaPipe Hand Detection
            if self.detector is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                try:
                    detection_result = self.detector.detect(mp_image)
                    if detection_result.hand_landmarks:
                        detected = True
                        landmarks = detection_result.hand_landmarks[0]
                        wrist = landmarks[0]
                        middle_mcp = landmarks[9]

                        # Palm coordinates
                        palm_nx_coord = 0.4 * wrist.x + 0.6 * middle_mcp.x
                        palm_ny_coord = 0.4 * wrist.y + 0.6 * middle_mcp.y

                        hand_scale = math.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)
                        hand_scale = max(0.06, min(0.35, hand_scale))

                        calib_x = np.clip((palm_nx_coord - 0.20) / 0.60, 0.0, 1.0)
                        calib_y = np.clip((palm_ny_coord - 0.18) / 0.64, 0.0, 1.0)

                        world_x = float((calib_x - 0.5) * 1.90)
                        world_y = float(0.04 + (1.0 - calib_y) * 0.51)

                        norm_s = float(np.clip((hand_scale - 0.08) / 0.14, 0.0, 1.0))
                        world_z = float(-1.55 + norm_s * 0.37)

                        # Biomechanical wrist & palm normal orientation
                        ux = middle_mcp.x - wrist.x
                        uy = middle_mcp.y - wrist.y
                        uz = getattr(middle_mcp, "z", 0.0) - getattr(wrist, "z", 0.0)

                        vx_palm = landmarks[17].x - landmarks[5].x
                        vy_palm = landmarks[17].y - landmarks[5].y
                        vz_palm = getattr(landmarks[17], "z", 0.0) - getattr(landmarks[5], "z", 0.0)

                        nx = uy * vz_palm - uz * vy_palm
                        ny = uz * vx_palm - ux * vz_palm
                        nz = ux * vy_palm - uy * vx_palm
                        n_norm = math.sqrt(nx * nx + ny * ny + nz * nz)
                        if n_norm > 1e-5:
                            palm_nx, palm_ny, palm_nz = nx / n_norm, ny / n_norm, nz / n_norm

                        roll = float(math.atan2(vy_palm, vx_palm))
                        pitch = float(-math.atan2(uy, math.sqrt(ux * ux + uz * uz) + 1e-6))
                        yaw = float(math.atan2(ux, uz + 1e-6))

                        # Detect closed fist
                        folded = 0
                        for tip_i, pip_i in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                            d_tip = math.hypot(landmarks[tip_i].x - wrist.x, landmarks[tip_i].y - wrist.y)
                            d_pip = math.hypot(landmarks[pip_i].x - wrist.x, landmarks[pip_i].y - wrist.y)
                            if d_tip < d_pip * 1.05:
                                folded += 1
                        is_fist = (folded >= 3)
                except Exception:
                    detected = False

            # 2. Resilient Motion Optical Centroid Fallback
            # If MediaPipe missed hand, use optical brightness centroid
            if not detected:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    M = cv2.moments(thresh)
                    if M["m00"] > 500:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        detected = True
                        world_x = float(((cx / w) - 0.5) * 1.90)
                        world_y = float(0.04 + (1.0 - (cy / h)) * 0.51)
                        world_z = -1.40
                prev_gray = gray

            # Compute velocities and accelerations
            vx = (world_x - prev_x) / dt
            vy = (world_y - prev_y) / dt
            vz = (world_z - prev_z) / dt
            speed = float(math.sqrt(vx * vx + vy * vy + vz * vz))
            speed_kmh = speed * 3.6

            ax = (vx - prev_vx) / dt
            ay = (vy - prev_vy) / dt
            az = (vz - prev_vz) / dt

            prev_x, prev_y, prev_z = world_x, world_y, world_z
            prev_vx, prev_vy, prev_vz = vx, vy, vz

            rec = FrameKinematics(
                frame_index=frame_idx,
                timestamp_sec=ts,
                detected=detected,
                world_x=world_x,
                world_y=world_y,
                world_z=world_z,
                vx=vx,
                vy=vy,
                vz=vz,
                speed=speed,
                speed_kmh=speed_kmh,
                ax=ax,
                ay=ay,
                az=az,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                palm_nx=palm_nx,
                palm_ny=palm_ny,
                palm_nz=palm_nz,
                hand_scale=hand_scale,
                is_fist=is_fist,
                stroke_label=stroke_label,
                video_source=os.path.basename(video_path),
            )
            records.append(asdict(rec))

        cap.release()
        return pd.DataFrame(records)
