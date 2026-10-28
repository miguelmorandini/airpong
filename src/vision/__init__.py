"""Computer vision package for hand tracking and gesture recognition."""

from src.vision.camera import CameraSource, WebcamSource, SyntheticCameraSource
from src.vision.gestures import GestureClassifier, GestureResult
from src.vision.tracker import HandTracker, HandTrackingResult

__all__ = [
    "CameraSource",
    "WebcamSource",
    "SyntheticCameraSource",
    "GestureClassifier",
    "GestureResult",
    "HandTracker",
    "HandTrackingResult",
]
