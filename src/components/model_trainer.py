# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/components/model_trainer.py
# Description: This module handles the model training process.
#              It orchestrates data loading, transformation, model building, and training.
# Author: LALAN KUMAR
# Created: [08-01-2025]
# Updated: [02-05-2025]
# Last Modified By: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import os
import sys
import numpy as np
import tensorflow as tf
from dataclasses import dataclass

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import load_yaml, train_model # Assuming train_model handles saving
# Import model building functions (adjust paths/names as needed)
from src.models.cnn_model import build_cnn_model
from src.models.ae_model import build_ae_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str
    model_name: str
    learning_rate: float
    epochs: int
    patch_size: int

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.trainer_config = config
        logging.info(f"ModelTrainer initialized for model: {self.trainer_config.model_name}")

    def initiate_model_training(self, train_dataset, test_dataset, num_features, num_classes):
        """Builds and trains the model using preprocessed data."""
        logging.info("Initiating model building and training...")
        try:
            # --- 3. Model Building ---
            logging.info("--- Step 3: Model Building ---")
            
            # Input shape uses patch_size from the config object
            input_shape = (self.trainer_config.patch_size, self.trainer_config.patch_size, num_features)
            logging.info(f"Using input shape: {input_shape}, Num features: {num_features}, Num classes: {num_classes}")

            model = None
            # Use model_name from the config object
            if self.trainer_config.model_name == 'cnn':
                if num_classes <= 1:
                     raise ValueError("Cannot train CNN: Not enough classes found in labels.")
                logging.info(f"Building CNN model.")
                model = build_cnn_model(
                    in_channels=num_features, 
                    n_classes=num_classes, 
                    learning_rate=self.trainer_config.learning_rate
                )
            elif self.trainer_config.model_name == 'ae':
                 # AE might still use num_classes for a potential classification layer
                logging.info(f"Building Autoencoder model.")
                model = build_ae_model(
                    in_channels=num_features, 
                    n_classes=num_classes, # AE might use this for a classification head
                    # Use learning_rate from the config object
                    learning_rate=self.trainer_config.learning_rate 
                )
            else:
                # Use model_name from the config object
                raise ValueError(f"Unsupported model type: {self.trainer_config.model_name}")
            
            if model is None:
                raise RuntimeError("Model building failed.")
            # Model is compiled within the build functions
            model.summary(print_fn=logging.info) # Log model summary

            # --- 4. Model Training ---
            logging.info("--- Step 4: Model Training ---")
            history = train_model(
                model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                # Use save path from the config object
                save_model_path=self.trainer_config.trained_model_file_path,
                # Use epochs from the config object
                epochs=self.trainer_config.epochs
            )
            # Use save path from the config object
            logging.info(f"Model training completed. Model saved to {self.trainer_config.trained_model_file_path}")
            final_val_accuracy = history.history.get('val_accuracy', [None])[-1]
            final_val_loss = history.history.get('val_loss', [None])[-1]
            logging.info(f"Final validation loss: {final_val_loss}")
            logging.info(f"Final validation accuracy: {final_val_accuracy}")

            logging.info("Model training process finished successfully.")
             # Use save path from the config object
            return self.trainer_config.trained_model_file_path

        except ValueError as e:
            logging.error(f"Configuration or data validation error during model build/train: {e}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.exception(f"An unexpected error occurred during model build/train: {e}")
            raise CustomException(e, sys)

# Example usage:
if __name__ == "__main__":
    logging.info("Starting model training script execution (with integrated data processing)...")
    try:
        # Load full configuration
        CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))
        config = load_yaml(CONFIG_FILE)
        logging.info("YAML Configuration loaded.")

        # --- Extract General Configs ---
        data_dir = config.get('data_dir')
        if not data_dir:
             raise ValueError("Missing 'data_dir' in top-level config.")
        dataset_name = config['model_trainer']['dataset']
        if not dataset_name:
             raise ValueError("Missing 'dataset_name' in 'model_trainer' config.")
        
        # --- 1. Data Ingestion ---
        logging.info("--- Step 1: Data Ingestion ---")
        logging.info(f"Starting Data Ingestion for dataset: {dataset_name} from {data_dir}")
        data_ingestion = DataIngestion(data_dir=data_dir, config_file=CONFIG_FILE)
        images, labels = data_ingestion.initiate_data_ingestion(dataset_name=dataset_name)
        if images is None or labels is None:
            raise ValueError("Data ingestion failed, returned None.")
        logging.info(f"Data Ingestion completed. Image shape: {images.shape}, Labels shape: {labels.shape}")
        original_bands = images.shape[-1] # Store original bands before potential PCA

        # --- 2. Data Transformation --- 
        # Create DataTransformationConfig object first
        logging.info("--- Step 2a: Creating DataTransformationConfig ---")
        artifacts_root = config['transformation']['artifacts_root']
        if not artifacts_root:
             raise ValueError("Missing 'artifacts_root' in 'transformation' config")
             
        transformation_config_obj = DataTransformationConfig(
            artifacts_root=artifacts_root,
            dataset_name=dataset_name,
            data_dir=data_dir,
            preprocessor_file_path=config['transformation']['preprocessor_file_path'],
            n_components=config['transformation']['pca_components'], # Use pca_components from yaml
            patch_size=config['transformation']['patch_size'],
            test_size=config['transformation']['test_size'],
            batch_size=config['transformation']['batch_size']
        )
        # Basic validation for required fields in transformation_config_obj
        if not transformation_config_obj.preprocessor_file_path or not transformation_config_obj.patch_size:
             raise ValueError("Missing 'preprocessor_file_path' or 'patch_size' in transformation config")

        logging.info("--- Step 2b: Running Data Transformation ---")
        data_transformation = DataTransformation(config=transformation_config_obj) # Pass the object
        train_dataset, test_dataset, transformer, X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(images, labels)
        logging.info("Data Transformation successful.")

        # --- 3. Determine Features and Classes ---
        logging.info("--- Step 3: Determining Model Inputs ---")
        if data_transformation.apply_pca:
            num_features = transformation_config_obj.n_components
            logging.info(f"PCA applied. Using {num_features} features.")
        else:
            num_features = original_bands
            logging.info(f"PCA not applied. Using {num_features} features.")
            
        num_classes = len(np.unique(y_train))
        logging.info(f"Found {num_classes} unique classes.")

        # --- 4. Prepare ModelTrainer Config ---
        logging.info("--- Step 4: Preparing ModelTrainerConfig ---")
        model_name = config['model_trainer']['model'].lower()
        model_filename = f"{model_name}_model.keras" 
        model_save_path = config['model_trainer']['model_save_path']
        trained_model_file_path = os.path.join(model_save_path, model_filename)
        
        # --- Ensure the target directory for the model exists ---
        model_dir = os.path.dirname(model_save_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create the ModelTrainerConfig object
        trainer_config_obj = ModelTrainerConfig(
            trained_model_file_path=trained_model_file_path,
            model_name=model_name,
            learning_rate=config['model_trainer']['learning_rate'],
            epochs=config['model_trainer']['epochs'],
            patch_size=transformation_config_obj.patch_size
        )

        # --- 5. Instantiate and Run Model Trainer ---
        logging.info("--- Step 5: Initializing and Running ModelTrainer ---")
        trainer = ModelTrainer(config=trainer_config_obj) # Pass the object
        
        trained_model_path = trainer.initiate_model_training(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_features=num_features,
            num_classes=num_classes
        )
        
        if trained_model_path:
            print(f" Model training complete. Model saved at: {trained_model_path}")
        else:
            print(" Model training failed.")
            
    except FileNotFoundError as e:
        logging.error(f"Configuration or data file not found: {e}")
        print(f" Error: File not found - {e}")
    except ValueError as e:
        logging.error(f"Configuration or data validation error: {e}")
        print(f" Error: Invalid Configuration or Data - {e}")
    except CustomException as e:
        logging.error(f"Custom Exception: {e}")
        print(f" An error occurred: {e}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during script execution: {e}")
        print(f" An unexpected error occurred: {e}")


