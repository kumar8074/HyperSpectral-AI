# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/pipeline/predict_pipeline.py
# Description: Orchestrates the complete Prediction pipeline for both CNN and AutoEncoder models
# Author: LALAN KUMAR
# Created: [07-01-2025]
# Updated: [14-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import os
import sys
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Union
import tensorflow as tf
from tensorflow.keras.models import load_model

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.exception import CustomException
from src.logger import logging
from src.utils import (
    load_yaml,
    get_predictions,
    calculate_metrics,
    get_classification_report,
    plot_confusion_matrix,
    visualize_predictions
)
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.models.autoencoder_model import HyperspectralAE


@dataclass
class PredictionConfig:
    """
    Configuration class for prediction pipeline parameters
    """
    model_type: str = "cnn"
    config_path: str = os.path.join(project_root, 'config/config.yaml')
    visualize_results: bool = True
    save_report: bool = False

class PredictionPipeline:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = None
        self.transformer = None
        self.label_values = []
        self._initialize_components()

    def _initialize_components(self):
        """Load configuration, model and transformer"""
        try:
            # Load base configuration
            self.base_config = load_yaml(self.config.config_path)
            
            # Validate model type
            if self.config.model_type.lower() not in ["cnn", "ae"]:
                raise ValueError(f"Invalid model_type: {self.config.model_type}. Must be 'cnn' or 'ae'.")
            
            # Set model and transformer paths
            model_type = self.config.model_type.lower()
            self.model_path = self.base_config['model_trainer'][f'model_file_path_{model_type}']
            self.transformer_path = self.base_config['transformation']['preprocessor_file_path'].replace(
                '.pkl', f'_{model_type}.pkl')
            
            # Load components
            self.model = load_model(self.model_path)
            self.transformer = DataTransformation.load_transformer(self.transformer_path)
            
            logging.info(f"Successfully initialized {model_type.upper()} prediction pipeline")

        except Exception as e:
            logging.error("Component initialization failed", exc_info=True)
            raise CustomException(e, sys)

    def _get_dataset_config(self, dataset_name: str) -> Dict:
        """Retrieve dataset-specific configuration"""
        for dataset in self.base_config['datasets']:
            if dataset['name'] == dataset_name:
                self.label_values = dataset['label_values']
                return dataset
        raise ValueError(f"Dataset {dataset_name} not found in configuration")

    def prepare_data(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Handle complete data preparation pipeline"""
        try:
            # Data Ingestion
            data_ingestion = DataIngestion(
                data_dir=self.base_config['data_dir'],
                config_file=self.config.config_path
            )
            images, labels = data_ingestion.initiate_data_ingestion(dataset_name)
            
            # Data Transformation
            transformation_config = DataTransformationConfig(
                pca_components=self.base_config['transformation']['pca_components'],
                patch_size=self.base_config['transformation']['patch_size'],
                test_size=self.base_config['transformation']['test_size'],
                batch_size=self.base_config['transformation']['batch_size'],
                transformer_obj_file_path=self.base_config['transformation']['preprocessor_file_path'],
                use_pca=(self.config.model_type == "cnn")
            )
            
            data_transformation = DataTransformation(config=transformation_config)
            _, test_dataset, _ = data_transformation.initiate_data_transformation(images, labels)
            
            return test_dataset, labels

        except Exception as e:
            logging.error("Data preparation failed", exc_info=True)
            raise CustomException(e, sys)

    def evaluate(self, test_dataset, label_image, dataset_name: str) -> Dict[str, Union[dict, str]]:
        """Perform complete evaluation pipeline"""
        try:
            # Get predictions
            y_true, y_pred, unique_classes, filtered_labels = get_predictions(
                self.model, test_dataset, self.label_values, ae=(self.config.model_type == "ae")
            )
            
            # Calculate metrics
            metrics = calculate_metrics(y_true, y_pred)
            report = get_classification_report(y_true, y_pred, unique_classes, filtered_labels)
            
            # Visualization
            if self.config.visualize_results:
                plot_confusion_matrix(y_true, y_pred, unique_classes, filtered_labels)
                visualize_predictions(y_true, y_pred, title="True vs Pred")
            
            return {
                'metrics': metrics,
                'report': report,
                'true_labels': y_true,
                'pred_labels': y_pred
            }

        except Exception as e:
            logging.error("Evaluation failed", exc_info=True)
            raise CustomException(e, sys)

    def run_pipeline(self, dataset_name: str):
        """Execute complete prediction pipeline"""
        try:
            # Validate dataset
            self._get_dataset_config(dataset_name)
            
            # Prepare data
            test_dataset, label_image = self.prepare_data(dataset_name)
            
            # Perform evaluation
            results = self.evaluate(test_dataset, label_image=label_image, dataset_name=dataset_name)
            
            # Print results
            print(f"\n{'='*40} Results {'='*40}")
            print(f"Model Type: {self.config.model_type.upper()}")
            print(f"Dataset: {dataset_name}")
            print("\n📈 Metrics:")
            for metric, value in results['metrics'].items():
                print(f"{metric.capitalize()}: {value:.4f}")
            
            print("\n📝 Classification Report:")
            print(results['report'])
            
            return results

        except Exception as e:
            logging.error("Prediction pipeline failed", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Hyperspectral Image Classification Prediction Pipeline")
        parser.add_argument("--model", type=str, choices=["cnn", "ae"], default="cnn",
                          help="Model type to use for prediction")
        parser.add_argument("--dataset", type=str, required=True,
                          help="Name of the dataset to process (must exist in config.yaml)")
        parser.add_argument("--no-vis", action="store_false", dest="visualize",
                          help="Disable visualization of results")
        
        args = parser.parse_args()
        
        # Initialize prediction pipeline
        pred_config = PredictionConfig(
            model_type=args.model,
            visualize_results=args.visualize
        )
        
        pipeline = PredictionPipeline(pred_config)
        results = pipeline.run_pipeline(args.dataset)
        
        print("\n✅ Prediction pipeline completed successfully")

    except Exception as e:
        print(f"\n❌ Prediction pipeline failed: {str(e)}")
        sys.exit(1)


