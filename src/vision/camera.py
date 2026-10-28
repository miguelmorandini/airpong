"""Camera abstractions for real webcams and synthetic/mock cameras."""

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraSource(ABC):
    """Abstract base class for video capture sources."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame (BGR format)."""
        pass

    @abstractmethod
    def release(self):
        """Release camera resources."""
        pass

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera source is active."""
        pass


class WebcamSource(CameraSource):
    """Real physical webcam capture via OpenCV."""

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.width = width
        self.height = height
        # Use DirectShow on Windows for fastest startup
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(device_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, 60)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Flip horizontally for natural mirror feel (like Wii Sports)
            frame = cv2.flip(frame, 1)
        return ret, frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()


class SyntheticCameraSource(CameraSource):
    """
    Synthetic camera generator for testing and headless CI.
    Generates realistic video frames with an animated virtual hand.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 60):
        self.width = width
        self.height = height
        self.fps = fps
        self._is_active = True
        self._start_time = time.time()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._is_active:
            return False, None

        t = time.time() - self._start_time
        # Create dark blue background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (30, 25, 20)  # Dark slate

        # Animate hand palm circle oscillating left and right
        cx = int(self.width * (0.5 + 0.3 * np.sin(t * 2.5)))
        cy = int(self.height * (0.6 + 0.15 * np.cos(t * 3.0)))
        radius = int(35 + 8 * np.sin(t * 1.5))

        # Draw skin-colored circle and fingers to simulate a hand
        skin_color = (180, 200, 240)  # BGR skin tone
        cv2.circle(frame, (cx, cy), radius, skin_color, -1)
        # 5 fingers
        for i in range(5):
            angle = -np.pi / 2 + (i - 2) * 0.25
            fx = int(cx + (radius + 25) * np.cos(angle))
            fy = int(cy + (radius + 25) * np.sin(angle))
            cv2.line(frame, (cx, cy), (fx, fy), skin_color, 8)
            cv2.circle(frame, (fx, fy), 6, skin_color, -1)

        cv2.putText(
            frame,
            "SYNTHETIC CAMERA SOURCE",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 100),
            2,
        )

        return True, frame

    def release(self):
        self._is_active = False

    @property
    def is_opened(self) -> bool:
        return self._is_active
