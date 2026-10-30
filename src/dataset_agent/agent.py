"""Table Tennis Motion & Hand Tracking Dataset Agent Orchestrator."""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.dataset_agent.video_miner import TableTennisVideoMiner, STROKE_SEARCH_PRESETS
from src.dataset_agent.hand_extractor import HandKinematicsExtractor
from src.dataset_agent.dataset_builder import DatasetBuilder
from src.dataset_agent.model_trainer import StrokeModelTrainer


class TableTennisDatasetAgent:
    """Autonomous Agent for sourcing ping pong movement videos, extracting hand landmarks,

    and training ML gesture classifiers.
    """

    def __init__(
        self,
        raw_video_dir: str = "data/raw_videos",
        processed_dir: str = "data/processed",
        model_dir: str = "src/vision/models",
    ):
        self.raw_video_dir = raw_video_dir
        self.processed_dir = processed_dir
        self.model_dir = model_dir

        self.miner = TableTennisVideoMiner(storage_dir=self.raw_video_dir)
        self.extractor = HandKinematicsExtractor()
        self.builder = DatasetBuilder(
            raw_video_dir=self.raw_video_dir,
            processed_dir=self.processed_dir,
            extractor=self.extractor,
        )
        self.trainer = StrokeModelTrainer(model_dir=self.model_dir)

    def fetch_videos(
        self,
        strokes: Optional[List[str]] = None,
        count_per_stroke: int = 1,
        custom_query: Optional[str] = None,
    ) -> List[str]:
        """Fetch videos from YouTube or generate synthetic biomechanical clips."""
        downloaded = []
        if custom_query:
            print(f"[Agent] Buscando vídeos com query personalizada: '{custom_query}'...")
            paths = self.miner.search_and_download(query=custom_query, max_results=count_per_stroke)
            downloaded.extend(paths)
            return downloaded

        target_strokes = strokes or list(STROKE_SEARCH_PRESETS.keys())
        for stroke in target_strokes:
            print(f"[Agent] Obtendo vídeos para golpe: {stroke}...")
            paths = self.miner.download_by_preset(stroke_label=stroke, count=count_per_stroke)
            downloaded.extend(paths)

        print(f"[Agent] Total de vídeos adquiridos: {len(downloaded)}")
        return downloaded

    def extract_and_build(self, use_augmented_fallback: bool = True) -> pd.DataFrame:
        """Extract landmarks from all downloaded videos and build dataset."""
        print("[Agent] Iniciando extração cinemática dos frames via MediaPipe...")
        df = self.builder.process_all_raw_videos()
        if df.empty and use_augmented_fallback:
            print("[Agent] Gerando dataset aumentado biomecânico de tênis de mesa...")
            df = self.builder.generate_augmented_dataset(samples_per_class=120)

        parquet_path, csv_path, json_path = self.builder.save_dataset(df)
        print(f"[Agent] Dataset consolidado com {len(df)} amostras salvas em {csv_path}")
        return df

    def train_model(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train and evaluate the gesture classifier."""
        if df is None:
            df = self.builder.load_or_create_dataset()

        print(f"[Agent] Treinando classificador de movimentos com {len(df)} amostras...")
        metrics = self.trainer.train(df)
        print(f"[Agent] Modelo treinado com sucesso! Acurácia: {metrics['accuracy_percentage']}%")
        return metrics

    def run_full_pipeline(self, fetch_count: int = 1) -> Dict[str, Any]:
        """Execute end-to-end pipeline: Fetch Videos -> Extract Kinematics -> Build Dataset -> Train ML."""
        print("=" * 70)
        print("[AGENT] INICIANDO PIPELINE DE TREINAMENTO DE MOVIMENTOS DE TENIS DE MESA")
        print("=" * 70)

        # 1. Video Sourcing
        videos = self.fetch_videos(count_per_stroke=fetch_count)

        # 2. Extract & Build
        df = self.extract_and_build()

        # 3. Train Model
        metrics = self.train_model(df)

        print("=" * 70)
        print("[AGENT] PIPELINE CONCLUIDO COM SUCESSO!")
        print(f"[METRICAS] Acuracia Geral: {metrics['accuracy_percentage']}%")
        print(f"[CLASSES] Classes Treinadas: {', '.join(metrics['classes'])}")
        print(f"[SAVED] Modelo Salvo em: {self.trainer.model_path}")
        print("=" * 70)

        return {
            "videos_count": len(videos),
            "samples_count": len(df),
            "metrics": metrics,
        }


def main():
    parser = argparse.ArgumentParser(description="Table Tennis Movement Video & Dataset Agent")
    subparsers = parser.add_subparsers(dest="command")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch or generate videos")
    fetch_parser.add_argument("--stroke", type=str, default=None, help="Stroke label")
    fetch_parser.add_argument("--count", type=int, default=1, help="Count of videos per stroke")
    fetch_parser.add_argument("--query", type=str, default=None, help="Custom search query")

    # Build command
    subparsers.add_parser("build", help="Extract landmarks and build dataset")

    # Train command
    subparsers.add_parser("train", help="Train model from dataset")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline end-to-end")
    pipeline_parser.add_argument("--count", type=int, default=1, help="Videos per stroke")

    args = parser.parse_args()
    agent = TableTennisDatasetAgent()

    if args.command == "fetch":
        strokes = [args.stroke] if args.stroke else None
        agent.fetch_videos(strokes=strokes, count_per_stroke=args.count, custom_query=args.query)
    elif args.command == "build":
        agent.extract_and_build()
    elif args.command == "train":
        agent.train_model()
    elif args.command == "pipeline" or args.command is None:
        count = getattr(args, "count", 1) if args.command else 1
        agent.run_full_pipeline(fetch_count=count)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
