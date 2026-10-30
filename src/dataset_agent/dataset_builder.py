"""Dataset aggregation, labeling, and export for table tennis hand gestures."""

import os
import math
import json
import glob
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.dataset_agent.hand_extractor import HandKinematicsExtractor
from src.dataset_agent.video_miner import TableTennisVideoMiner, STROKE_SEARCH_PRESETS


FEATURE_COLUMNS = [
    "vx",
    "vy",
    "vz",
    "speed",
    "ax",
    "ay",
    "az",
    "roll",
    "pitch",
    "yaw",
    "palm_nx",
    "palm_ny",
    "palm_nz",
    "hand_scale",
    "is_fist",
]

TARGET_COLUMN = "stroke_label"


class DatasetBuilder:
    """Aggregates raw kinematic extractions into normalized datasets for machine learning."""

    def __init__(
        self,
        raw_video_dir: str = "data/raw_videos",
        processed_dir: str = "data/processed",
        extractor: Optional[HandKinematicsExtractor] = None,
    ):
        self.raw_video_dir = raw_video_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.extractor = extractor or HandKinematicsExtractor()

    def process_all_raw_videos(self, include_augmented: bool = True) -> pd.DataFrame:
        """Process all videos in raw_video_dir and compile into a single DataFrame."""
        extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
        video_files = []
        for ext in extensions:
            video_files.extend(glob.glob(os.path.join(self.raw_video_dir, ext)))

        if not video_files:
            print("[DatasetBuilder] Nenhum vídeo encontrado em data/raw_videos. Gerando vídeos sintéticos de treino...")
            miner = TableTennisVideoMiner(storage_dir=self.raw_video_dir)
            for stroke in STROKE_SEARCH_PRESETS.keys():
                miner.generate_synthetic_video(stroke_type=stroke)
            for ext in extensions:
                video_files.extend(glob.glob(os.path.join(self.raw_video_dir, ext)))

        dfs = []
        for v_path in video_files:
            fname = os.path.basename(v_path).upper()
            # Infer stroke label from filename
            stroke_label = "IDLE"
            for candidate in ["GANCHO_FOREHAND", "GANCHO_BACKHAND", "GANCHO_SMASH", "CHOP", "SERVE"]:
                if candidate in fname:
                    stroke_label = candidate
                    break

            print(f"[DatasetBuilder] Processando vídeo: {os.path.basename(v_path)} (Label: {stroke_label})...")
            df = self.extractor.process_video(v_path, stroke_label=stroke_label)
            if not df.empty:
                dfs.append(df)

        if include_augmented:
            augmented_df = self.generate_augmented_dataset(samples_per_class=100)
            dfs.append(augmented_df)

        if not dfs:
            print("[DatasetBuilder] Aviso: nenhum frame extraído dos vídeos. Gerando dataset aumentado biomecânico...")
            return self.generate_augmented_dataset(samples_per_class=150)

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def generate_augmented_dataset(self, samples_per_class: int = 150) -> pd.DataFrame:
        """
        Synthesizes an augmented biomechanically accurate kinematic dataset.
        Models real player arm speed, wrist angles, and paddle velocity
        according to ITTF table tennis dynamics.
        """
        np.random.seed(42)
        records = []

        stroke_definitions = {
            "IDLE": {
                "vx": (0.0, 0.25),
                "vy": (0.0, 0.20),
                "vz": (0.0, 0.20),
                "roll": (0.0, 0.15),
                "pitch": (0.0, 0.15),
                "yaw": (0.0, 0.15),
                "fist_prob": 0.05,
            },
            "GANCHO_FOREHAND": {
                # Right-handed forehand sweep: positive vx (sweep to right/center), upward lift
                "vx": (2.6, 0.8),
                "vy": (0.7, 0.4),
                "vz": (0.5, 0.3),
                "roll": (0.45, 0.2),   # closed blade topspin
                "pitch": (-0.35, 0.2), # forward tilt
                "yaw": (0.30, 0.25),
                "fist_prob": 0.0,
            },
            "GANCHO_BACKHAND": {
                # Right-handed backhand: negative vx (sweep to left/exterior), upward flick
                "vx": (-2.4, 0.75),
                "vy": (0.6, 0.4),
                "vz": (0.6, 0.35),
                "roll": (-0.40, 0.2),
                "pitch": (-0.25, 0.2),
                "yaw": (-0.35, 0.2),
                "fist_prob": 0.0,
            },
            "GANCHO_SMASH": {
                # Extreme downward/forward snap, high speed
                "vx": (0.4, 0.6),
                "vy": (-2.4, 0.7),     # strong downward velocity
                "vz": (3.8, 0.9),      # deep forward snap
                "roll": (0.2, 0.3),
                "pitch": (-0.8, 0.25), # aggressive steep blade tilt
                "yaw": (0.1, 0.2),
                "fist_prob": 0.0,
            },
            "CHOP": {
                # Defensive slice: downward movement with open blade (positive pitch)
                "vx": (0.3, 0.4),
                "vy": (-1.6, 0.4),
                "vz": (-0.3, 0.3),
                "roll": (0.1, 0.2),
                "pitch": (0.65, 0.2),  # open blade for backspin
                "yaw": (0.0, 0.2),
                "fist_prob": 0.0,
            },
            "SERVE": {
                # Pendulum serve: moderate forward speed, gentle lateral motion, distinct wrist cocking
                "vx": (0.3, 0.3),
                "vy": (0.35, 0.25),
                "vz": (1.3, 0.35),
                "roll": (0.7, 0.25),   # pronounced wrist rotation
                "pitch": (0.1, 0.2),
                "yaw": (0.2, 0.2),
                "fist_prob": 0.0,
            },
        }

        for label, params in stroke_definitions.items():
            for i in range(samples_per_class):
                vx = float(np.random.normal(params["vx"][0], params["vx"][1]))
                vy = float(np.random.normal(params["vy"][0], params["vy"][1]))
                vz = float(np.random.normal(params["vz"][0], params["vz"][1]))
                speed = float(math.sqrt(vx * vx + vy * vy + vz * vz))

                # Accelerations correlated with velocity
                dt = 0.033
                ax = float(vx / dt + np.random.normal(0, 3.0))
                ay = float(vy / dt + np.random.normal(0, 3.0))
                az = float(vz / dt + np.random.normal(0, 3.0))

                roll = float(np.random.normal(params["roll"][0], params["roll"][1]))
                pitch = float(np.random.normal(params["pitch"][0], params["pitch"][1]))
                yaw = float(np.random.normal(params["yaw"][0], params["yaw"][1]))

                palm_nx = float(np.sin(yaw))
                palm_ny = float(-np.sin(pitch))
                palm_nz = float(np.cos(pitch) * np.cos(yaw))
                norm = math.sqrt(palm_nx**2 + palm_ny**2 + palm_nz**2) or 1.0
                palm_nx /= norm
                palm_ny /= norm
                palm_nz /= norm

                hand_scale = float(np.random.uniform(0.12, 0.22))
                is_fist = bool(np.random.rand() < params["fist_prob"])

                records.append({
                    "vx": vx,
                    "vy": vy,
                    "vz": vz,
                    "speed": speed,
                    "speed_kmh": speed * 3.6,
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "palm_nx": palm_nx,
                    "palm_ny": palm_ny,
                    "palm_nz": palm_nz,
                    "hand_scale": hand_scale,
                    "is_fist": is_fist,
                    "stroke_label": label,
                    "video_source": "BIOMECHANICAL_AUGMENTED",
                })

        return pd.DataFrame(records)

    def save_dataset(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        """Save dataset to Parquet, CSV, and JSON formats."""
        parquet_path = os.path.join(self.processed_dir, "dataset_strokes.parquet")
        csv_path = os.path.join(self.processed_dir, "dataset_strokes.csv")
        json_path = os.path.join(self.processed_dir, "dataset_sequences.json")

        # Save Parquet
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            # Fallback if pyarrow/fastparquet is not present
            parquet_path = ""

        # Save CSV
        df.to_csv(csv_path, index=False)

        # Save sample JSON sequences
        sample_dict = df.head(100).to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sample_dict, f, indent=2)

        print(f"[DatasetBuilder] Dataset salvo com sucesso! {len(df)} amostras.")
        return parquet_path, csv_path, json_path

    def load_or_create_dataset(self) -> pd.DataFrame:
        """Load existing processed dataset or build one automatically."""
        csv_path = os.path.join(self.processed_dir, "dataset_strokes.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)

        df = self.process_all_raw_videos()
        self.save_dataset(df)
        return df
