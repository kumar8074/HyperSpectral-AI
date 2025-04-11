# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/pipeline/train_pipeline.py
# Description: Orchestrates the complete training pipeline for both CNN and AutoEncoder models
# Author: LALAN KUMAR
# Created: [12-01-2025]
# Updated: [15-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Literal

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.exception import CustomException
from src.logger import logging
from src.utils import load_yaml
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer_cnn import CNNModelTrainer, CNNModelTrainerConfig
from src.components.model_trainer_ae import AEModelTrainer, AEModelTrainerConfig

@dataclass
class TrainingPipelineConfig:
    config_file_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))

ModelType = Literal['cnn', 'ae', 'all']

class TrainingPipeline:
    def __init__(self, model_type: ModelType = 'all'):
        self.pipeline_config = TrainingPipelineConfig()
        self.config = load_yaml(self.pipeline_config.config_file_path)
        self.model_type = model_type.lower()
        self._validate_model_type()

    def _validate_model_type(self):
        """Validate the provided model type"""
        valid_types = ['cnn', 'ae', 'all']
        if self.model_type not in valid_types:
            raise ValueError(f"Invalid model type: {self.model_type}. Choose from {valid_types}")

    def _run_cnn_pipeline(self, images, labels, dataset_config):
        """Execute CNN training pipeline"""
        logging.info("Phase 2/4: CNN Training Pipeline Started")
        
        # Data Transformation
        cnn_transformation_config = DataTransformationConfig(
            pca_components=self.config['transformation']['pca_components'],
            patch_size=self.config['transformation']['patch_size'],
            test_size=self.config['transformation']['test_size'],
            batch_size=self.config['transformation']['batch_size'],
            transformer_obj_file_path=self.config['transformation']['preprocessor_file_path'],
            use_pca=True
        )
        data_transformer_cnn = DataTransformation(config=cnn_transformation_config)
        train_dataset_cnn, test_dataset_cnn, _ = data_transformer_cnn.initiate_data_transformation(images, labels)
        
        # Model Training
        cnn_trainer_config = CNNModelTrainerConfig(
            epochs=self.config['model_trainer']['epochs'],
            learning_rate=self.config['model_trainer']['learning_rate'],
            loss=self.config['model_trainer']['loss'],
            optimizer=self.config['model_trainer']['optimizer'],
            metrics=self.config['model_trainer']['metrics'],
            model_save_path=self.config['model_trainer']['model_file_path_cnn']
        )
        cnn_in_channels = self.config['transformation']['pca_components']
        cnn_trainer = CNNModelTrainer(config=cnn_trainer_config)
        cnn_trainer.initiate_model_training(
            train_dataset_cnn, test_dataset_cnn, 
            len(dataset_config['label_values']), cnn_in_channels
        )
        logging.info("✅ CNN Training Completed")

    def _run_ae_pipeline(self, images, labels, dataset_config):
        """Execute AutoEncoder training pipeline"""
        logging.info("Phase 3/4: AutoEncoder Training Pipeline Started")
        
        # Data Transformation
        ae_transformation_config = DataTransformationConfig(
            pca_components=self.config['transformation']['pca_components'],
            patch_size=self.config['transformation']['patch_size'],
            test_size=self.config['transformation']['test_size'],
            batch_size=self.config['transformation']['batch_size'],
            transformer_obj_file_path=self.config['transformation']['preprocessor_file_path'],
            use_pca=False
        )
        data_transformer_ae = DataTransformation(config=ae_transformation_config)
        train_dataset_ae, test_dataset_ae, _ = data_transformer_ae.initiate_data_transformation(images, labels)
        
        # Model Training
        ae_trainer_config = AEModelTrainerConfig(
            epochs=self.config['model_trainer']['epochs'],
            learning_rate=self.config['model_trainer']['learning_rate'],
            recon_loss=self.config['model_trainer']['recon_loss_ae'],
            class_loss=self.config['model_trainer']['class_loss_ae'],
            recon_weight=self.config['model_trainer']['recon_weight_ae'],
            class_weight=self.config['model_trainer']['class_weight_ae'],
            optimizer=self.config['model_trainer']['optimizer'],
            metrics=self.config['model_trainer']['metrics'],
            model_save_path=self.config['model_trainer']['model_file_path_ae']
        )
        ae_in_channels = images.shape[-1]
        ae_trainer = AEModelTrainer(config=ae_trainer_config)
        ae_trainer.initiate_model_training(
            train_dataset_ae, test_dataset_ae,
            len(dataset_config['label_values']), ae_in_channels
        )
        logging.info("✅ AutoEncoder Training Completed")

    def run_pipeline(self):
        """Execute the complete training pipeline based on selected model"""
        try:
            logging.info("Starting hyperspectral training pipeline")

            # ==================== Data Ingestion ====================
            logging.info("Phase 1/4: Data Ingestion Started")
            data_ingestion = DataIngestion(
                data_dir=self.config['data_dir'],
                config_file=self.pipeline_config.config_file_path
            )
            images, labels = data_ingestion.initiate_data_ingestion(
                dataset_name=self.config['model_trainer']['dataset']
            )
            logging.info("✅ Data Ingestion Completed")

            # ==================== Dataset Configuration ====================
            dataset_name = self.config['model_trainer']['dataset']
            dataset_config = next(
                (ds for ds in self.config['datasets'] if ds['name'] == dataset_name),
                None
            )
            if not dataset_config:
                raise ValueError(f"Dataset {dataset_name} not found in configuration")

            # ==================== Model Training ====================
            if self.model_type in ['cnn', 'all']:
                self._run_cnn_pipeline(images, labels, dataset_config)
            
            if self.model_type in ['ae', 'all']:
                self._run_ae_pipeline(images, labels, dataset_config)

            logging.info("Phase 4/4: Pipeline Completion")
            logging.info("🚀 Training pipeline completed successfully")
            print("✅ Training pipeline completed successfully")

        except Exception as e:
            logging.error(f"❌ Pipeline execution failed: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Setup command line arguments
        parser = argparse.ArgumentParser(description='Hyperspectral Model Training Pipeline')
        parser.add_argument('--model', type=str.lower, 
                          choices=['cnn', 'ae', 'all'], 
                          default='ae',
                          help='Select model to train: cnn, ae, or all (default)')
        args = parser.parse_args()

        # Execute pipeline with selected model
        pipeline = TrainingPipeline(model_type=args.model)
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"Pipeline execution error: {e}")
        print(f"Error: {e}")