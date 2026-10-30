"""Unit tests for Table Tennis Dataset Agent, Video Miner, Extractor, and Model Trainer."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from src.dataset_agent.video_miner import TableTennisVideoMiner
from src.dataset_agent.hand_extractor import HandKinematicsExtractor
from src.dataset_agent.dataset_builder import DatasetBuilder, FEATURE_COLUMNS, TARGET_COLUMN
from src.dataset_agent.model_trainer import StrokeModelTrainer
from src.dataset_agent.agent import TableTennisDatasetAgent
from src.vision.gestures import GestureClassifier


def test_video_miner_synthetic_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = TableTennisVideoMiner(storage_dir=tmpdir)
        vid_path = miner.generate_synthetic_video(
            stroke_type="GANCHO_FOREHAND",
            duration_sec=0.5,
            fps=20,
        )
        assert os.path.exists(vid_path)
        meta = miner.get_video_metadata(vid_path, stroke_label="GANCHO_FOREHAND")
        assert meta is not None
        assert meta.frame_count >= 10
        assert meta.fps == 20.0
        assert meta.stroke_label == "GANCHO_FOREHAND"


def test_hand_kinematics_extractor():
    with tempfile.TemporaryDirectory() as tmpdir:
        miner = TableTennisVideoMiner(storage_dir=tmpdir)
        vid_path = miner.generate_synthetic_video(
            stroke_type="GANCHO_BACKHAND",
            duration_sec=0.5,
            fps=20,
        )

        extractor = HandKinematicsExtractor()
        df = extractor.process_video(vid_path, stroke_label="GANCHO_BACKHAND")
        assert not df.empty
        assert len(df) >= 10
        assert "speed" in df.columns
        assert "vx" in df.columns
        assert "vy" in df.columns
        assert "vz" in df.columns
        assert "roll" in df.columns
        assert "pitch" in df.columns
        assert "yaw" in df.columns


def test_dataset_builder_augmented_and_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = os.path.join(tmpdir, "raw")
        proc_dir = os.path.join(tmpdir, "processed")
        builder = DatasetBuilder(raw_video_dir=raw_dir, processed_dir=proc_dir)

        df = builder.generate_augmented_dataset(samples_per_class=30)
        assert len(df) >= 150
        assert TARGET_COLUMN in df.columns
        for feat in FEATURE_COLUMNS:
            assert feat in df.columns

        pq, csv, js = builder.save_dataset(df)
        assert os.path.exists(csv)
        assert os.path.exists(js)


def test_stroke_model_trainer_and_prediction():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "models")
        trainer = StrokeModelTrainer(model_dir=model_dir)
        builder = DatasetBuilder(processed_dir=tmpdir)
        df = builder.generate_augmented_dataset(samples_per_class=40)

        metrics = trainer.train(df, test_size=0.25)
        assert metrics["accuracy"] >= 0.85
        assert "GANCHO_FOREHAND" in metrics["classes"]
        assert "GANCHO_BACKHAND" in metrics["classes"]
        assert "GANCHO_SMASH" in metrics["classes"]
        assert os.path.exists(trainer.model_path)
        assert os.path.exists(trainer.meta_path)

        # Predict single sample
        sample_feat = {
            "vx": 2.5,
            "vy": 0.5,
            "vz": 0.5,
            "speed": 2.6,
            "ax": 10.0,
            "ay": 2.0,
            "az": 2.0,
            "roll": 0.4,
            "pitch": -0.3,
            "yaw": 0.2,
            "palm_nx": 0.2,
            "palm_ny": 0.3,
            "palm_nz": 0.9,
            "hand_scale": 0.16,
            "is_fist": 0,
        }
        pred_label, conf = trainer.predict(sample_feat)
        assert pred_label == "GANCHO_FOREHAND"
        assert conf > 0.50


def test_agent_integration_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = os.path.join(tmpdir, "raw")
        proc_dir = os.path.join(tmpdir, "processed")
        model_dir = os.path.join(tmpdir, "models")

        agent = TableTennisDatasetAgent(
            raw_video_dir=raw_dir,
            processed_dir=proc_dir,
            model_dir=model_dir,
        )

        miner = TableTennisVideoMiner(storage_dir=raw_dir)
        miner.generate_synthetic_video(stroke_type="GANCHO_FOREHAND", duration_sec=0.3, fps=15)
        miner.generate_synthetic_video(stroke_type="GANCHO_BACKHAND", duration_sec=0.3, fps=15)

        df = agent.extract_and_build(use_augmented_fallback=True)
        assert len(df) > 0
        metrics = agent.train_model(df)
        assert metrics["accuracy"] > 0.80
