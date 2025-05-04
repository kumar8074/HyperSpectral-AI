# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/pipeline/train_pipeline.py
# Description: This file defines the main training pipeline for the project.
# Author: LALAN KUMAR
# Created: [09-01-2025]
# Updated: [03-05-2025]
# Last Modified By: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

"""Main training pipeline script for the Hyperspectral AI project.

Orchestrates the data ingestion, data transformation, and model training steps
based on the provided configuration file.
"""

import os
import sys
import argparse
import numpy as np

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import load_yaml

class TrainPipeline:
    """Manages the execution of the entire training workflow."""
    def __init__(self, config_path: str):
        """Initializes the pipeline by loading configurations.

        Args:
            config_path (str): Path to the main YAML configuration file.
        """
        self.config_path = config_path
        self.config = None
        self.data_ingestion_config = None
        self.data_transformation_config = None
        self.model_trainer_config = None
        self.load_configs()

    def load_configs(self):
        """Loads configurations from the main YAML file."""
        try:
            logging.info(f"Loading configuration from: {self.config_path}")
            self.config = load_yaml(self.config_path)
            if not self.config:
                raise ValueError("Configuration file is empty or could not be loaded.")

            # --- Extract General Configs ---
            data_dir = self.config.get('data_dir')
            if not data_dir:
                 raise ValueError("Missing 'data_dir' in top-level config.")
            dataset_name = self.config['model_trainer']['dataset']
            if not dataset_name:
                 raise ValueError("Missing 'dataset' in 'model_trainer' config.")
            artifacts_root = self.config['transformation']['artifacts_root']
            if not artifacts_root:
                 raise ValueError("Missing 'artifacts_root' in 'transformation' config")

            # --- Create DataTransformationConfig ---
            self.data_transformation_config = DataTransformationConfig(
                artifacts_root=artifacts_root,
                dataset_name=dataset_name,
                data_dir=data_dir,
                preprocessor_file_path=self.config['transformation']['preprocessor_file_path'],
                n_components=self.config['transformation']['pca_components'],
                patch_size=self.config['transformation']['patch_size'],
                test_size=self.config['transformation']['test_size'],
                batch_size=self.config['transformation']['batch_size']
            )
            if not self.data_transformation_config.preprocessor_file_path or not self.data_transformation_config.patch_size:
                 raise ValueError("Missing 'preprocessor_file_path' or 'patch_size' in transformation config")

            # --- Create ModelTrainerConfig ---
            model_name = self.config['model_trainer']['model'].lower()
            model_save_path = self.config['model_trainer']['model_save_path']
            model_filename = f"{model_name}_model.keras" # Consistent naming
            trained_model_file_path = os.path.join(model_save_path, model_filename)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(trained_model_file_path), exist_ok=True)
            
            self.model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=trained_model_file_path,
                model_name=model_name,
                learning_rate=self.config['model_trainer']['learning_rate'],
                epochs=self.config['model_trainer']['epochs'],
                patch_size=self.data_transformation_config.patch_size # Use patch size from transform config
            )
            logging.info("Configurations loaded successfully.")

        except KeyError as e:
            logging.error(f"Missing key in configuration file: {e}")
            raise CustomException(f"Missing key in configuration file: {e}", sys)
        except Exception as e:
            logging.error(f"Error loading configurations: {e}")
            raise CustomException(e, sys)

    def run_pipeline(self):
        """
        Executes the full training pipeline: ingestion, transformation, and model training.
        Returns:
            str: Path to the trained model file
        """
        logging.info("[TrainPipeline] Starting the training pipeline...")
        try:
            # Step 1: Data Ingestion
            logging.info("[TrainPipeline] Step 1: Data Ingestion")
            data_ingestion = DataIngestion(
                data_dir=self.data_transformation_config.data_dir,
                config_file=self.config_path
            )
            images, labels = data_ingestion.initiate_data_ingestion(self.data_transformation_config.dataset_name)
            if images is None or labels is None:
                raise ValueError("Data ingestion failed, returned None.")
            
            # Step 2: Data Transformation
            logging.info("[TrainPipeline] Step 2: Data Transformation")
            data_transformation = DataTransformation(self.data_transformation_config)
            train_dataset, test_dataset, X_train, X_test, y_train, y_test, transformer = data_transformation.initiate_data_transformation(images, labels)
            
            # Calculate number of features
            if hasattr(data_transformation, 'apply_pca') and data_transformation.apply_pca:
                num_features = self.data_transformation_config.n_components
                logging.info(f"PCA applied. Using {num_features} features.")
            else:
                num_features = images.shape[-1]  # Original number of bands
                logging.info(f"PCA not applied. Using {num_features} features.")
            
            # Compute number of classes robustly
            label_flat = labels.flatten()
            label_flat = label_flat[label_flat != 0]  # Remove background class if present
            num_classes = len(np.unique(label_flat))
            logging.info(f"Found {num_classes} unique classes in the labels.")
            
            # Validate class count for classification models
            if num_classes <= 1 and self.model_trainer_config.model_name != 'ae':
                raise ValueError(f"Only {num_classes} class found. Need at least 2 classes for classification models.")

            # Step 3: Model Training
            logging.info("[TrainPipeline] Step 3: Model Training")
            model_trainer = ModelTrainer(self.model_trainer_config)
            trained_model_path = model_trainer.initiate_model_training(
                train_dataset, test_dataset, num_features, num_classes
            )
            
            logging.info(f"Model training finished. Trained model saved at: {trained_model_path}")
            logging.info("===== Training Pipeline Completed Successfully ====")
            
            return trained_model_path

        except CustomException as ce:
            logging.error(f"Error during training pipeline: {ce}")
            raise ce  # Re-raise the custom exception
        except Exception as e:
            logging.error(f"An unexpected error occurred in the training pipeline: {e}")
            raise CustomException(e, sys)

# Entry point for running the pipeline directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Hyperspectral AI Training Pipeline.")
    parser.add_argument("--config", type=str, 
                        default=os.path.join(project_root, 'config', 'config.yaml'),
                        help="Path to the main configuration YAML file.")
    args = parser.parse_args()

    try:
        pipeline = TrainPipeline(config_path=args.config)
        trained_model_path = pipeline.run_pipeline()
        print(f"Pipeline completed successfully. Model saved at: {trained_model_path}")
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)  # Exit with a non-zero code to indicate failure
    
    
