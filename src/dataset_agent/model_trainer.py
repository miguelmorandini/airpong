"""Model training and evaluation module for table tennis hand gestures."""

import os
import json
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset_agent.dataset_builder import FEATURE_COLUMNS, TARGET_COLUMN


DEFAULT_MODEL_DIR = os.path.join("src", "vision", "models")
MODEL_FILENAME = "stroke_classifier.joblib"
METADATA_FILENAME = "stroke_model_meta.json"


class StrokeModelTrainer:
    """Trains, evaluates, and exports ML classifiers for table tennis stroke gestures."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, MODEL_FILENAME)
        self.meta_path = os.path.join(self.model_dir, METADATA_FILENAME)

        self.model: Optional[RandomForestClassifier] = None
        self.metadata: Dict[str, Any] = {}

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Train Random Forest classifier on extracted kinematics features."""
        # Ensure all required features are present
        features = [col for col in FEATURE_COLUMNS if col in df.columns]
        X = df[features].copy()

        # Handle boolean or missing values
        if "is_fist" in X.columns:
            X["is_fist"] = X["is_fist"].astype(int)
        X = X.fillna(0.0)

        y = df[TARGET_COLUMN].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=75,
            max_depth=12,
            min_samples_split=3,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # Evaluation
        y_pred = clf.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        labels = sorted(list(set(y)))
        cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

        # Feature importances
        importances = {
            feat: float(imp)
            for feat, imp in zip(features, clf.feature_importances_)
        }
        sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

        self.model = clf
        self.metadata = {
            "accuracy": accuracy,
            "accuracy_percentage": round(accuracy * 100, 2),
            "classes": labels,
            "features": features,
            "feature_importances": sorted_importances,
            "confusion_matrix": cm,
            "classification_report": report,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save model and metadata
        self._save()
        print(f"[StrokeModelTrainer] Treinamento concluído com Acurácia de {accuracy * 100:.2f}%!")
        return self.metadata

    def _save(self):
        """Save trained model and metadata to disk."""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)

    def load(self) -> bool:
        """Load trained model and metadata from disk."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                if os.path.exists(self.meta_path):
                    with open(self.meta_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                return True
            except Exception as e:
                print(f"[StrokeModelTrainer] Erro ao carregar modelo: {e}")
                return False
        return False

    def predict(self, feature_dict: Dict[str, float]) -> Tuple[str, float]:
        """Predict stroke type and confidence from a single feature vector."""
        if self.model is None and not self.load():
            return "IDLE", 0.0

        features = self.metadata.get("features", FEATURE_COLUMNS)
        vec = [float(feature_dict.get(f, 0.0)) for f in features]
        x_in = np.array([vec], dtype=np.float32)

        pred_label = self.model.predict(x_in)[0]
        probs = self.model.predict_proba(x_in)[0]
        pred_idx = list(self.model.classes_).index(pred_label)
        confidence = float(probs[pred_idx])

        return pred_label, confidence
