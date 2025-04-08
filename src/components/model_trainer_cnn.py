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
from src.utils import get_optimizer, get_loss, get_metric, train_model, load_yaml  # Reusable utilities
from src.models.cnn_model import HyperspectralCNN  # CNN Model definition
from src.components.data_transformation import DataTransformation, DataTransformationConfig  # Data transformation
from src.components.data_ingestion import DataIngestion  # Data ingestion

@dataclass
class CNNModelTrainerConfig:
    """
    Configuration class for defining parameters used during CNN model training.
    """
    epochs: int  # Number of epochs to train
    learning_rate: float  # Learning rate for the optimizer
    loss: str  # Loss function to use
    optimizer: str  # Optimizer to use
    metrics: list  # Metrics to track during training
    model_save_path: str  # Path to save the trained model


class CNNModelTrainer:
    def __init__(self, config: CNNModelTrainerConfig):
        self.trainer_config = config

    def train(self, train_dataset, test_dataset, n_classes, in_channels):
        """
        Train the CNN model using the provided datasets and configuration.

        Args:
            train_dataset: TensorFlow dataset for training.
            test_dataset: TensorFlow dataset for testing.
            n_classes: Number of output classes.
            in_channels: Number of input channels (e.g., number of PCA components).

        Returns:
            model: The trained TensorFlow Keras model.
        """
        try:
            # Validate dataset compatibility
            for data, label in train_dataset.take(1):
                input_shape = data.shape[1:]
                if input_shape[-1] != in_channels:
                    raise ValueError(f"Mismatch in input channels. Expected: {in_channels}, Got: {input_shape[-1]}")
                
            # Initialize the CNN model
            model = HyperspectralCNN(in_channels=in_channels, n_classes=n_classes)
            logging.info("CNN model initialized with in_channels: %s, n_classes: %s", in_channels, n_classes)

            # Compile the model
            optimizer = get_optimizer(self.trainer_config.optimizer)(learning_rate=self.trainer_config.learning_rate)
            loss = get_loss(self.trainer_config.loss)
            metrics = [get_metric(metric) for metric in self.trainer_config.metrics]

            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            logging.info("CNN model compiled with optimizer: %s, loss: %s, metrics: %s",
                         self.trainer_config.optimizer, self.trainer_config.loss, self.trainer_config.metrics)
            
            # Ensure model save directory exists
            model_dir = os.path.dirname(self.trainer_config.model_save_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                logging.info(f"Model directory created at: {model_dir}")

            # Check for None values in the datasets
            if train_dataset is None or test_dataset is None:
                raise ValueError("Train or test dataset contains None values.")
            
            # Train the model
            logging.info("Starting CNN model training")
            train_model(
                model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                save_model_path=self.trainer_config.model_save_path,
                epochs=self.trainer_config.epochs
            )
            logging.info("CNN model training completed.")

            return model

        except Exception as e:
            logging.error("Error during CNN model training: %s", str(e))
            raise CustomException(e, sys)
            
    def initiate_model_training(self, train_dataset, test_dataset, n_classes, in_channels):
        """
        Initiates the complete model training process including validation, model configuration,
        compilation, and training.
        
        Args:
            train_dataset: TensorFlow dataset for training.
            test_dataset: TensorFlow dataset for testing.
            n_classes: Number of output classes.
            in_channels: Number of input channels (e.g., number of PCA components).
            
        Returns:
            model: The trained TensorFlow Keras model.
        """
        logging.info("Starting model training process")
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
            logging.info("Model training process completed successfully")
            
            return model
            
        except Exception as e:
            logging.error(f"Error in model training process: {e}")
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

        # Data transformation configurations
        transformation_config = DataTransformationConfig(
            pca_components=config['transformation']['pca_components'],
            patch_size=config['transformation']['patch_size'],
            test_size=config['transformation']['test_size'],
            batch_size=config['transformation']['batch_size'],
            transformer_obj_file_path=config['transformation']['preprocessor_file_path'],
            use_pca=config['model_trainer']['use_pca']
        )

        # Data transformation
        data_transformation = DataTransformation(config=transformation_config)
        train_dataset, test_dataset, transformer = data_transformation.initiate_data_transformation(images, labels)

        # Get the number of classes and input channels
        n_classes = len(config['datasets'][dataset_index]['label_values'])
        in_channels = config['transformation']['pca_components'] if config['model_trainer']['use_pca'] else images.shape[-1]

        # CNN model trainer configurations
        trainer_config = CNNModelTrainerConfig(
            epochs=config['model_trainer']['epochs'],
            learning_rate=config['model_trainer']['learning_rate'],
            loss=config['model_trainer']['loss'],
            optimizer=config['model_trainer']['optimizer'],
            metrics=config['model_trainer']['metrics'],
            model_save_path=config['model_trainer']['model_file_path_cnn']
        )

        # Model training using the new initiate_model_training method
        model_trainer = CNNModelTrainer(config=trainer_config)
        model = model_trainer.initiate_model_training(train_dataset, test_dataset, n_classes, in_channels)

        logging.info("CNN model training completed successfully.")
        print("✅ CNN model training pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error in CNN model training script: {e}")
        raise CustomException(e, sys)