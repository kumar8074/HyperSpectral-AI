# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/pipeline/predict_pipeline.py
# Description: This file defines the main prediction pipeline for the project.
# Author: LALAN KUMAR
# Created: [09-01-2025]
# Updated: [03-05-2025]
# Last Modified By: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

"""Prediction pipeline script for the Hyperspectral AI project.

Loads a trained model and preprocessor, performs inference on a dataset,
calculates evaluation metrics, and saves results and visualizations.
"""


import os
import sys
import argparse
import time
import warnings
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.exceptions import UndefinedMetricWarning

# Suppress undefined metric warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Add project root to path
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, os.pardir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.components.data_ingestion import DataIngestion
from src.utils import (
    load_yaml,
    apply_pca,
    extract_patches,
    plot_confusion_matrix,
    plot_label_comparison_spatial
)
from src.logger import logging
from src.models.cnn_model import build_cnn_model
from src.models.ae_model import build_ae_model, HyperspectralAE


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


class PredictPipeline:
    """Pipeline for hyperspectral image model prediction and evaluation."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load_configs()

    def _load_configs(self) -> None:
        logging.info(f"[START] Loading config from {self.config_path}")
        cfg = load_yaml(self.config_path)
        if not cfg:
            raise ValueError("Empty or invalid config file.")

        # Paths and general settings
        self.data_dir = cfg['data_dir']
        trainer_cfg = cfg['model_trainer']
        transf_cfg = cfg['transformation']

        # Dataset and model type
        self.dataset_name = trainer_cfg['dataset']
        self.model_type = trainer_cfg['model'].lower()

        # PCA settings
        self.n_components = transf_cfg.get('pca_components')

        # Patch extraction settings
        self.patch_size = transf_cfg['patch_size']
        self.normalize_per_patch = transf_cfg.get('normalize_per_patch', True)
        self.standardize_patches = transf_cfg.get('standardize_patches', False)
        self.batch_size = transf_cfg.get('batch_size', 32)

        # Model paths
        model_dir = trainer_cfg['model_save_path']
        model_file = 'cnn_model.keras' if self.model_type == 'cnn' else 'ae_model.keras'
        self.model_path = os.path.join(model_dir, model_file)

        # Output directories
        self.results_dir = os.path.join('artifacts', 'results', self.model_type)
        self.vis_dir = os.path.join('artifacts', 'visualizations', self.model_type)
        ensure_dir(self.results_dir)
        ensure_dir(self.vis_dir)

        logging.info(f"Config loaded: model={self.model_type}, dataset={self.dataset_name}")
        logging.info(f"Model path: {self.model_path}")

    def reconstruct_image(self, preds: np.ndarray, locations: list, shape: tuple) -> np.ndarray:
        """Rebuild label map from patch predictions."""
        img = np.zeros(shape[:2], dtype=preds.dtype)
        if len(preds) != len(locations):
            raise ValueError("Predictions and locations count mismatch.")
        for label, (r, c) in zip(preds, locations):
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                img[r, c] = label
        return img

    def run_prediction(self):
        """Execute prediction workflow end-to-end."""
        t_start = time.time()
        logging.info("[Predict] Beginning pipeline")

        # 1. Data ingestion
        di = DataIngestion(self.data_dir, self.config_path)
        images, labels = di.initiate_data_ingestion(self.dataset_name)

        # 2. Load model
        logging.info(f"Loading {self.model_type.upper()} model")
        custom_objs = {'HyperspectralAE': HyperspectralAE} if self.model_type == 'ae' else {}
        model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objs)
        logging.info(f"Model '{model.name}' loaded")

        # 3. Preprocess
        proc = images
        if self.model_type == 'cnn' and self.n_components:
            logging.info(f"Applying PCA ({self.n_components} components)")
            proc = apply_pca(images, n_components=self.n_components)

        # normalize overall
        max_val = proc.max()
        if max_val > 1.0:
            proc = proc / max_val
            logging.info(f"Normalized data by {max_val}")

        # 4. Extract patches
        X_test, y_test, locs = extract_patches(
            proc,
            labels,
            patch_size=self.patch_size,
            normalize_per_patch=self.normalize_per_patch,
            standardize_patches=self.standardize_patches
        )
        logging.info(f"Extracted {X_test.shape[0]} patches")

        # 5. Predict
        y_probs = model.predict(X_test, batch_size=self.batch_size)
        if self.model_type == 'cnn':
            y_pred = np.argmax(y_probs, axis=1) + 1
        else:
            y_pred = np.argmax(y_probs[1] if isinstance(y_probs, (list, tuple)) else y_probs, axis=1) + 1
        logging.info("Prediction complete")

        # 6. Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        logging.info(f"Metrics -- Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

        # Save metrics
        metrics_file = os.path.join(self.results_dir, f"{self.dataset_name}_metrics.txt")
        with open(metrics_file, 'w') as mf:
            mf.write(f"Metrics -- Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}\n")
            mf.write("\n")
            mf.write("classification report:\n")
            mf.write(report)
        logging.info(f"Saved metrics to {metrics_file}")

        # 7. Reconstruct and visualize
        pred_img = self.reconstruct_image(y_pred, locs, labels.shape)
        # Confusion matrix
        cm_path = os.path.join(self.vis_dir, f"{self.dataset_name}_confusion_matrix.png")
        plot_confusion_matrix(y_test, y_pred, labels=np.unique(y_test), save_path=cm_path)
        logging.info(f"Saved confusion matrix to {cm_path}")
        # Spatial comparison
        comp_path = os.path.join(self.vis_dir, f"{self.dataset_name}_spatial.png")
        plot_label_comparison_spatial(y_test, y_pred, labels.shape, comp_path, len(np.unique(y_test)), labels_flat=labels.flatten())
        logging.info(f"Saved spatial comparison to {comp_path}")

        logging.info(f"Pipeline completed in {time.time() - t_start:.2f}s")
        return y_pred, acc, cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Hyperspectral Prediction Pipeline")
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(project_root, 'config', 'config.yaml'),
        help='Path to config YAML'
    )
    args = parser.parse_args()

    try:
        pipeline = PredictPipeline(args.config)
        pipeline.run_prediction()
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)










