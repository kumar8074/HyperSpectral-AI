# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/components/model_trainer_ae.py
# Description: This module handles the AutoEncoder model training process for hyperspectral data.
#              It includes data ingestion, transformation, model configuration, compilation, and training.
# Author: LALAN KUMAR
# Created: [11-01-2025]
# Updated: [10-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import os
import sys
import tensorflow as tf
from dataclasses import dataclass

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Logging setup
from src.utils import get_optimizer, get_loss, load_yaml  # Reusable utilities
from src.models.autoencoder_model import HyperspectralAE  # Autoencoder Model definition
from src.components.data_transformation import DataTransformation, DataTransformationConfig  # Data transformation
from src.components.data_ingestion import DataIngestion  # Data ingestion

@dataclass
class AEModelTrainerConfig:
    """
    Configuration class for defining parameters used during Autoencoder model training.
    """
    epochs: int  # Number of epochs to train
    learning_rate: float  # Learning rate for the optimizer
    recon_loss: str  # Loss function for reconstruction
    class_loss: str  # Loss function for classification
    recon_weight: float  # Weight for reconstruction loss
    class_weight: float  # Weight for classification loss
    optimizer: str  # Optimizer to use
    metrics: list  # Metrics to track during training
    model_save_path: str  # Path to save the trained model


class AEModelTrainer:
    def __init__(self, config: AEModelTrainerConfig):
        self.trainer_config = config

    def train(self, train_dataset, test_dataset, n_classes, in_channels):
        """
        Train the Autoencoder model using the provided datasets and configuration.

        Args:
            train_dataset: TensorFlow dataset for training.
            test_dataset: TensorFlow dataset for testing.
            n_classes: Number of output classes.
            in_channels: Number of input channels (spectral bands).

        Returns:
            model: The trained TensorFlow Keras model.
        """
        try:
            # Validate dataset compatibility
            for data, label in train_dataset.take(1):
                input_shape = data.shape[1:]
                if input_shape[-1] != in_channels:
                    raise ValueError(f"Mismatch in input channels. Expected: {in_channels}, Got: {input_shape[-1]}")
                
            # Initialize the Autoencoder model
            model = HyperspectralAE(in_channels=in_channels, n_classes=n_classes)
            logging.info("Autoencoder model initialized with in_channels: %s, n_classes: %s", in_channels, n_classes)

            # Get loss functions
            recon_loss = get_loss(self.trainer_config.recon_loss)
            class_loss = get_loss(self.trainer_config.class_loss)
            
            # Get optimizer
            optimizer = get_optimizer(self.trainer_config.optimizer)(learning_rate=self.trainer_config.learning_rate)
            
            # Compile the model with multiple outputs, losses, and weights
            model.compile(
                optimizer=optimizer,
                loss=[recon_loss, class_loss],  # Separate losses for reconstruction and classification
                loss_weights=[self.trainer_config.recon_weight, self.trainer_config.class_weight],
                metrics=['mse', 'accuracy']  # Metrics for each output
            )
            
            logging.info("Autoencoder model compiled with optimizer: %s, recon_loss: %s, class_loss: %s",
                         self.trainer_config.optimizer, self.trainer_config.recon_loss, self.trainer_config.class_loss)
            
            # Ensure model save directory exists
            model_dir = os.path.dirname(self.trainer_config.model_save_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                logging.info(f"Model directory created at: {model_dir}")

            # Check for None values in the datasets
            if train_dataset is None or test_dataset is None:
                raise ValueError("Train or test dataset contains None values.")
            
            # Define callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', 
                    patience=10, 
                    restore_best_weights=True,
                    mode='max'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.trainer_config.model_save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train the model
            logging.info("Starting Autoencoder model training")
            history = model.fit(
                train_dataset,
                validation_data=test_dataset,
                epochs=self.trainer_config.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            logging.info("Autoencoder model training completed.")
            return model

        except Exception as e:
            logging.error("Error during Autoencoder model training: %s", str(e))
            raise CustomException(e, sys)
            
    def initiate_model_training(self, train_dataset, test_dataset, n_classes, in_channels):
        """
        Initiates the complete model training process including validation, model configuration,
        compilation, and training.
        
        Args:
            train_dataset: TensorFlow dataset for training.
            test_dataset: TensorFlow dataset for testing.
            n_classes: Number of output classes.
            in_channels: Number of input channels (spectral bands).
            
        Returns:
            model: The trained TensorFlow Keras model.
        """
        logging.info("Starting Autoencoder model training process")
        try:
            # Validate inputs
            if train_dataset is None or test_dataset is None:
                raise ValueError("Train or test dataset cannot be None")
                
            if n_classes <= 0:
                raise ValueError(f"Invalid number of classes: {n_classes}")
                
            if in_channels <= 0:
                raise ValueError(f"Invalid number of input channels: {in_channels}")
                
            # Train the model
            model = self.train(train_dataset, test_dataset, n_classes, in_channels)
            
            logging.info(f"Model saved at: {self.trainer_config.model_save_path}")
            logging.info("Autoencoder model training process completed successfully")
            
            return model
            
        except Exception as e:
            logging.error(f"Error in Autoencoder model training process: {e}")
            raise CustomException(e, sys)


# Example usage as script
if __name__ == "__main__":
    try:
        # Paths and configurations
        CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))
        config = load_yaml(CONFIG_FILE)
        DATA_DIR = config['data_dir']
        DATASET_NAME = config['model_trainer']['dataset']

        # Data ingestion
        data_ingestion = DataIngestion(data_dir=DATA_DIR, config_file=CONFIG_FILE)
        images, labels = data_ingestion.initiate_data_ingestion(dataset_name=DATASET_NAME)

        # Find dataset index
        dataset_index = -1
        for i, dataset in enumerate(config['datasets']):
            if dataset['name'] == DATASET_NAME:
                dataset_index = i
                break
        
        if dataset_index == -1:
            raise ValueError(f"Dataset {DATASET_NAME} not found in configuration")

        # Data transformation configurations - Note: use_pca is set to False for Autoencoder
        transformation_config = DataTransformationConfig(
            pca_components=config['transformation']['pca_components'],
            patch_size=config['transformation']['patch_size'],
            test_size=config['transformation']['test_size'],
            batch_size=config['transformation']['batch_size'],
            transformer_obj_file_path=config['transformation']['preprocessor_file_path'],
            use_pca=False  # AutoEncoder workflow doesn't use PCA
        )

        # Data transformation
        data_transformation = DataTransformation(config=transformation_config)
        train_dataset, test_dataset, transformer = data_transformation.initiate_data_transformation(images, labels)

        # Get the number of classes and input channels
        n_classes = len(config['datasets'][dataset_index]['label_values'])
        in_channels = images.shape[-1]  # Original spectral bands (no PCA for autoencoder)

        # Autoencoder model trainer configurations
        trainer_config = AEModelTrainerConfig(
            epochs=config['model_trainer']['epochs'],
            learning_rate=config['model_trainer']['learning_rate'],
            recon_loss=config['model_trainer']['recon_loss_ae'],  # Default to MSE if not specified
            class_loss=config['model_trainer']['class_loss_ae'],
            recon_weight=config['model_trainer']['recon_weight_ae'],
            class_weight=config['model_trainer']['class_weight_ae'],
            optimizer=config['model_trainer']['optimizer'],
            metrics=config['model_trainer']['metrics'],
            model_save_path=config['model_trainer']['model_file_path_ae']
        )

        # Model training
        model_trainer = AEModelTrainer(config=trainer_config)
        model = model_trainer.initiate_model_training(train_dataset, test_dataset, n_classes, in_channels)

        logging.info("Autoencoder model training completed successfully.")
        print("✅ Autoencoder model training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error in Autoencoder model training script: {e}")
        raise CustomException(e, sys)



