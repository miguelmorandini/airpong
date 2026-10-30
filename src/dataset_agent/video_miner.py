"""Video mining and acquisition module for table tennis movements."""

import os
import re
import glob
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import cv2
import numpy as np

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False


# Curated high-yield search presets for table tennis strokes
STROKE_SEARCH_PRESETS: Dict[str, List[str]] = {
    "GANCHO_FOREHAND": [
        "table tennis forehand loop slow motion tutorial",
        "table tennis forehand drive hand wrist movement",
        "ping pong forehand topspin technique",
    ],
    "GANCHO_BACKHAND": [
        "table tennis backhand loop slow motion",
        "table tennis backhand flick wrist action",
        "table tennis backhand drive tutorial hand",
    ],
    "GANCHO_SMASH": [
        "table tennis smash slow motion technique",
        "table tennis power smash hit slow mo",
        "ping pong smash execution",
    ],
    "CHOP": [
        "table tennis chop defense slow motion",
        "table tennis backspin chop tutorial hand angle",
    ],
    "SERVE": [
        "table tennis pendulum serve slow motion wrist",
        "table tennis serve technique close up hand",
    ],
}


@dataclass
class VideoMetadata:
    video_path: str
    stroke_label: str
    source: str
    duration_sec: float
    fps: float
    frame_count: int
    resolution: tuple


class TableTennisVideoMiner:
    """Acquires, downloads, and generates table tennis video clips for motion training."""

    def __init__(self, storage_dir: str = "data/raw_videos"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def search_and_download(
        self,
        query: str,
        max_results: int = 1,
        stroke_label: str = "GANCHO_FOREHAND",
        max_duration_sec: int = 45,
    ) -> List[str]:
        """Search YouTube for table tennis videos and download top matching clips."""
        if not YT_DLP_AVAILABLE:
            print("[AVISO] yt-dlp não está instalado. Gerando vídeo de referência sintético...")
            synth = self.generate_synthetic_video(
                stroke_type=stroke_label,
                filename=f"synth_{stroke_label.lower()}_sample.mp4",
            )
            return [synth] if synth else []

        ydl_opts = {
            "format": "best/bestvideo[height<=720]+bestaudio/best[height<=720]/b",
            "outtmpl": os.path.join(self.storage_dir, f"{stroke_label}_%(id)s.%(ext)s"),
            "max_filesize": 40 * 1024 * 1024,  # 40MB limit
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "socket_timeout": 15,
        }

        downloaded_paths = []
        search_query = f"ytsearch{max_results}:{query}"

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_query, download=True)
                if not info:
                    return []
                entries = info.get("entries", [info]) if "entries" in info else [info]
                for entry in entries:
                    if not entry:
                        continue
                    video_id = entry.get("id", "")
                    expected_pattern = os.path.join(self.storage_dir, f"{stroke_label}_{video_id}.*")
                    matches = glob.glob(expected_pattern)
                    for m in matches:
                        if m.lower().endswith((".mp4", ".mkv", ".webm")):
                            downloaded_paths.append(os.path.abspath(m))
        except Exception as e:
            print(f"[AVISO] Erro ao baixar via yt-dlp ({e}). Gerando vídeo sintético para treino.")
            synth = self.generate_synthetic_video(
                stroke_type=stroke_label,
                filename=f"fallback_{stroke_label.lower()}_sample.mp4",
            )
            if synth:
                downloaded_paths.append(synth)

        # Fallback if download yielded nothing
        if not downloaded_paths:
            synth = self.generate_synthetic_video(
                stroke_type=stroke_label,
                filename=f"synth_{stroke_label.lower()}_sample.mp4",
            )
            if synth:
                downloaded_paths.append(synth)

        return downloaded_paths

    def download_by_preset(self, stroke_label: str, count: int = 1) -> List[str]:
        """Download videos for a specific stroke category using curated search queries."""
        queries = STROKE_SEARCH_PRESETS.get(stroke_label, [f"table tennis {stroke_label} technique"])
        query = queries[0]
        return self.search_and_download(query=query, max_results=count, stroke_label=stroke_label)

    def list_local_videos(self) -> List[str]:
        """List all video files in storage directory."""
        extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.storage_dir, ext)))
        return sorted(files)

    def get_video_metadata(self, video_path: str, stroke_label: str = "UNKNOWN") -> Optional[VideoMetadata]:
        """Extract duration, FPS, and dimensions from video file."""
        if not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()

        return VideoMetadata(
            video_path=video_path,
            stroke_label=stroke_label,
            source="LOCAL_MINED",
            duration_sec=duration,
            fps=fps,
            frame_count=frame_count,
            resolution=(w, h),
        )

    def generate_synthetic_video(
        self,
        stroke_type: str = "GANCHO_FOREHAND",
        filename: Optional[str] = None,
        duration_sec: float = 2.0,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
    ) -> str:
        """
        Generate a synthetic video rendering a stylized human hand performing
        table tennis strokes with authentic biomechanical trajectories.
        Guarantees instant offline training and validation.
        """
        if filename is None:
            filename = f"synthetic_{stroke_type.lower()}_{int(duration_sec*fps)}f.mp4"

        output_path = os.path.join(self.storage_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

        total_frames = int(duration_sec * fps)

        for i in range(total_frames):
            t = i / total_frames  # normalized time [0.0 .. 1.0]
            frame = np.full((height, width, 3), (25, 28, 35), dtype=np.uint8)

            # Table Tennis Background Grid
            cv2.line(frame, (0, int(height * 0.75)), (width, int(height * 0.75)), (60, 65, 80), 2)
            cv2.putText(
                frame,
                f"STROKE: {stroke_type} | FRAME: {i+1}/{total_frames}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 230, 150),
                2,
            )

            # Biomechanical trajectory depending on stroke type
            if stroke_type == "GANCHO_FOREHAND":
                # Sweep from bottom-left to top-right
                hx = int(width * (0.30 + 0.45 * t))
                hy = int(height * (0.65 - 0.35 * math.sin(t * math.pi)))
                wrist_angle = -20 + 55 * t
            elif stroke_type == "GANCHO_BACKHAND":
                # Sweep from center-right to high-left
                hx = int(width * (0.68 - 0.45 * t))
                hy = int(height * (0.62 - 0.30 * math.sin(t * math.pi)))
                wrist_angle = 30 - 60 * t
            elif stroke_type == "GANCHO_SMASH":
                # High wind-up, sharp downward slam
                if t < 0.4:
                    hx = int(width * (0.50 + 0.1 * t))
                    hy = int(height * (0.45 - 0.25 * (t / 0.4)))
                else:
                    progress = (t - 0.4) / 0.6
                    hx = int(width * 0.54)
                    hy = int(height * (0.20 + 0.55 * (progress ** 2)))
                wrist_angle = -45 + 90 * t
            elif stroke_type == "CHOP":
                # Downward high-to-low slice
                hx = int(width * (0.40 + 0.25 * t))
                hy = int(height * (0.25 + 0.45 * t))
                wrist_angle = 45 - 20 * t
            elif stroke_type == "SERVE":
                # Delicate pendulum arc
                hx = int(width * (0.45 + 0.15 * math.sin(t * 2 * math.pi)))
                hy = int(height * (0.50 + 0.10 * math.cos(t * 2 * math.pi)))
                wrist_angle = 15 * math.sin(t * 4 * math.pi)
            else:  # IDLE / DRIVE
                hx = int(width * (0.50 + 0.05 * math.sin(t * 2 * math.pi)))
                hy = int(height * (0.55 + 0.04 * math.cos(t * 2 * math.pi)))
                wrist_angle = 0

            # Render synthetic hand: palm + 5 finger rays + wrist
            palm_color = (220, 185, 160)
            finger_color = (190, 150, 130)
            joint_color = (0, 220, 220)

            # Draw arm/forearm
            arm_base_x = int(hx - 80 * math.cos(math.radians(wrist_angle + 90)))
            arm_base_y = int(hy + 120)
            cv2.line(frame, (arm_base_x, arm_base_y), (hx, hy), (100, 110, 140), 18)

            # Draw palm center
            cv2.circle(frame, (hx, hy), 28, palm_color, -1)
            cv2.circle(frame, (hx, hy), 28, (120, 90, 70), 2)

            # Draw wrist landmark
            cv2.circle(frame, (hx, hy + 22), 6, joint_color, -1)

            # Draw 5 fingers
            finger_offsets = [-35, -18, 0, 18, 35]
            lengths = [38, 48, 52, 46, 36]
            for deg_offset, length in zip(finger_offsets, lengths):
                rad = math.radians(wrist_angle - 90 + deg_offset)
                fx = int(hx + length * math.cos(rad))
                fy = int(hy + length * math.sin(rad))
                cv2.line(frame, (hx, hy), (fx, fy), finger_color, 7)
                cv2.circle(frame, (fx, fy), 5, joint_color, -1)

            # Paddle blade outline
            paddle_x = int(hx + 55 * math.cos(math.radians(wrist_angle - 90)))
            paddle_y = int(hy + 55 * math.sin(math.radians(wrist_angle - 90)))
            cv2.circle(frame, (paddle_x, paddle_y), 32, (30, 30, 200), -1)
            cv2.circle(frame, (paddle_x, paddle_y), 32, (255, 255, 255), 2)

            writer.write(frame)

        writer.release()
        return output_path
