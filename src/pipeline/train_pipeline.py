# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/pipeline/train_pipeline.py
# Description: Orchestrates the complete training pipeline for both CNN and AutoEncoder models
# Author: LALAN KUMAR
# Created: [12-01-2025]
# Updated: [12-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import os
import sys
import argparse

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Logging setup
from src.utils import load_yaml  # Reusable utilities
from src.components.data_ingestion import DataIngestion  # Data ingestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig  # Data transformation
from src.components.model_trainer_cnn import CNNModelTrainer, CNNModelTrainerConfig  # CNN Model training
from src.components.model_trainer_ae import AEModelTrainer, AEModelTrainerConfig  # Autoencoder Model training

def train_pipeline(model_type="cnn"):
    """
    Pipeline to orchestrate the training process for either CNN or Autoencoder models.
    
    Args:
        model_type (str): The type of model to train, either "cnn" or "ae" (autoencoder).
        
    Returns:
        None
    """
    try:
        # Paths and configurations
        CONFIG_FILE = os.path.abspath(os.path.join(project_root, 'config/config.yaml'))
        config = load_yaml(CONFIG_FILE)
        DATA_DIR = config['data_dir']
        DATASET_NAME = config['model_trainer']['dataset']  # Dynamically get the dataset name from the config
        
        # Use PCA for CNN but not for Autoencoder
        use_pca = True if model_type.lower() == "cnn" else False
        
        # Data ingestion
        logging.info(f"Starting data ingestion for dataset: {DATASET_NAME}")
        data_ingestion = DataIngestion(data_dir=DATA_DIR, config_file=CONFIG_FILE)
        images, labels = data_ingestion.initiate_data_ingestion(dataset_name=DATASET_NAME)
        logging.info(f"Successfully loaded dataset: {DATASET_NAME}")
        
        # Find dataset index and configuration
        dataset_index = -1
        for i, dataset in enumerate(config['datasets']):
            if dataset['name'] == DATASET_NAME:
                dataset_index = i
                break
        
        if dataset_index == -1:
            raise ValueError(f"Dataset {DATASET_NAME} not found in configuration")
            
        # Get the number of classes
        n_classes = len(config['datasets'][dataset_index]['label_values'])
        
        # Data transformation configurations
        transformation_config = DataTransformationConfig(
            pca_components=config['transformation']['pca_components'],
            patch_size=config['transformation']['patch_size'],
            test_size=config['transformation']['test_size'],
            batch_size=config['transformation']['batch_size'],
            transformer_obj_file_path=config['transformation']['preprocessor_file_path'],
            use_pca=use_pca
        )
        
        # Data transformation
        logging.info(f"Starting data transformation for {model_type.upper()} workflow")
        data_transformation = DataTransformation(config=transformation_config)
        train_dataset, test_dataset, transformer = data_transformation.initiate_data_transformation(images, labels)
        logging.info(f"Data transformation completed for {model_type.upper()} workflow")
        
        # Set input channels based on model type
        if model_type.lower() == "cnn":
            # For CNN, use PCA components as input channels
            in_channels = config['transformation']['pca_components'] 
            
            # CNN model trainer configurations
            trainer_config = CNNModelTrainerConfig(
                epochs=config['model_trainer']['epochs'],
                learning_rate=config['model_trainer']['learning_rate'],
                loss=config['model_trainer']['loss'],
                optimizer=config['model_trainer']['optimizer'],
                metrics=config['model_trainer']['metrics'],
                model_save_path=config['model_trainer']['model_file_path_cnn']
            )
            
            # CNN model training
            logging.info("Starting CNN model training")
            model_trainer = CNNModelTrainer(config=trainer_config)
            model = model_trainer.initiate_model_training(train_dataset, test_dataset, n_classes, in_channels)
            logging.info("CNN model training completed successfully")
            
        else:  # Autoencoder workflow
            # For AE, use original spectral bands as input channels
            in_channels = images.shape[-1]  
            
            # Autoencoder model trainer configurations
            trainer_config = AEModelTrainerConfig(
                epochs=config['model_trainer']['epochs'],
                learning_rate=config['model_trainer']['learning_rate'],
                recon_loss=config['model_trainer']['recon_loss_ae'],
                class_loss=config['model_trainer']['class_loss_ae'],
                recon_weight=config['model_trainer']['recon_weight_ae'],
                class_weight=config['model_trainer']['class_weight_ae'],
                optimizer=config['model_trainer']['optimizer'],
                metrics=config['model_trainer']['metrics'],
                model_save_path=config['model_trainer']['model_file_path_ae']
            )
            
            # Autoencoder model training
            logging.info("Starting Autoencoder model training")
            model_trainer = AEModelTrainer(config=trainer_config)
            model = model_trainer.initiate_model_training(train_dataset, test_dataset, n_classes, in_channels)
            logging.info("Autoencoder model training completed successfully")
        
        
            
        logging.info(f"{model_type.upper()} model training pipeline completed successfully")
        print(f"✅ {model_type.upper()} model training pipeline completed successfully")
            
    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
        print(f"❌ Error in training pipeline: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error in training pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for hyperspectral image classification")
    parser.add_argument("--model", type=str, choices=["cnn", "ae"], default="cnn",
                       help="Type of model to train: 'cnn' for CNN model or 'ae' for Autoencoder model")
    args = parser.parse_args()
    
    train_pipeline(model_type=args.model)