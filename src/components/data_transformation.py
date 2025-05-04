# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/components/data_transformation.py
# Description: This file handles the data transformation process for hyperspectral images.
#              It includes dimensionality reduction, patch extraction and splitting into train and test sets for specified HSI data.
# Author: LALAN KUMAR
# Created: [07-01-2025]
# Updated: [02-05-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

"""Data Transformation component for preprocessing hyperspectral data.

Handles dimensionality reduction (PCA) based on model type, patch extraction,
data normalization, and splitting data into training and testing sets.
Includes a Transformer class to manage PCA state and can be run standalone.
"""

import os
import sys
from dataclasses import dataclass

# Dynamically add the project root directory to sys.path
# Allows importing modules from the 'src' directory
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Use absolute imports again
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Logging setup
from src.utils import (
    load_yaml, apply_pca, extract_patches,
    preprocess_and_split, save_object, load_transformer,
    normalize_patches
)  # Reusable utilities

@dataclass
class DataTransformationConfig:
    """
    Configuration class for defining parameters used during data transformation.
    """
    # Non-default fields first
    artifacts_root: str
    dataset_name: str
    data_dir: str
    preprocessor_file_path: str
    # Default fields after
    n_components: int = 30  # Default PCA components
    patch_size: int = 5     # Default patch size
    test_size: float = 0.2  # Fraction of data to use for testing
    batch_size: int = 32    # Batch size for TensorFlow datasets
    random_state: int = 42  # Seed for reproducibility

class Transformer:
    """
    A class to encapsulate the PCA and patch size for transforming input data.
    Handles both PCA and non-PCA (AE) workflows.
    """
    def __init__(self, pca, patch_size):
        self.pca = pca
        self.patch_size = patch_size

    def transform_input_data(self, input_data):
        """
        Transform input data using PCA (if available), else pass-through.
        Args:
            input_data (np.ndarray): The data to transform (e.g., H, W, C or patches).
        Returns:
            np.ndarray: Transformed data.
        """
        try:
            if self.pca is None:
                logging.info("[Transformer] Skipping PCA transformation (AE workflow). Returning input as-is.")
                return input_data
                
            logging.info("[Transformer] Applying PCA transformation to input data (CNN workflow).")
            original_shape = input_data.shape
            
            if input_data.ndim == 3:
                reshaped_data = input_data.reshape(-1, original_shape[-1])
            elif input_data.ndim == 2:
                reshaped_data = input_data
            else:
                raise ValueError("Unsupported input shape for PCA transformation.")

            # Apply PCA transform
            transformed_data = self.pca.transform(reshaped_data)
            logging.info("Input data transformed successfully using PCA.")

            # Reshape back if original was 3D
            if input_data.ndim == 3:
                transformed_data = transformed_data.reshape(*original_shape[:-1], -1)

            return transformed_data

        except Exception as e:
            logging.error(f"[Transformer] Error during input transformation: {str(e)}")
            raise CustomException(e, sys)

class DataTransformation:
    """Orchestrates the data transformation pipeline for hyperspectral data."""
    def __init__(self, config: DataTransformationConfig):
        self.transformation_config = config
        # Load the full config to check the model type
        config_path = os.path.join(project_root, 'config', 'config.yaml')
        full_config = load_yaml(config_path)
        model_name = full_config['model_trainer']['model'].lower()
        # Determine if PCA should be applied based on model
        self.apply_pca = (model_name != 'ae')
        if self.apply_pca:
            logging.info(f"Model type '{model_name}' requires PCA. PCA will be applied.")
        else:
            logging.info(f"Model type '{model_name}' (AutoEncoder) does not require PCA. Skipping PCA.")

    def get_transformer_object(self, images):
        """
        Create a data transformer object containing PCA and patch extraction configurations.
        """
        try:
            if not self.apply_pca:
                logging.info("Skipping PCA fitting as per model type (AutoEncoder).")
                transformer = Transformer(pca=None, patch_size=self.transformation_config.patch_size)
            else:
                logging.info("Applying PCA as per model type.")
                pca = PCA(n_components=self.transformation_config.n_components)
                reshaped_images = images.reshape(-1, images.shape[-1])
                pca.fit(reshaped_images)
                logging.info("PCA fitted on training data successfully.")
                transformer = Transformer(pca=pca, patch_size=self.transformation_config.patch_size)

            return transformer
        except Exception as e:
            logging.error(f"Error occurred while creating transformer object: {e}")
            raise CustomException(e, sys)

    def transform_data(self, images, labels=None):
        """
        Apply data transformation including optional PCA and patch extraction.
        """
        try:
            if self.apply_pca:
                logging.info("[DataTransformation] Applying PCA to input images.")
                images = apply_pca(images, self.transformation_config.n_components)
            else:
                logging.info("[DataTransformation] Skipping PCA (AutoEncoder workflow).")
                # For AE, input images remain unchanged

            # Extract patches with per-patch normalization and optional standardization
            patches, valid_labels, _ = extract_patches(
                images,
                labels,
                patch_size=self.transformation_config.patch_size,
                normalize_per_patch=True,
                standardize_patches=self.apply_pca
            )
            logging.info(f"[DataTransformation] Patch extraction complete. Patches shape: {patches.shape}")

            return patches, valid_labels
        except Exception as e:
            logging.error(f"[DataTransformation] Error during data transformation: {str(e)}")
            raise CustomException(e, sys)

    def prepare_datasets(self, patches, valid_labels):
        """
        Preprocess the transformed data and split into training and testing datasets.
        """
        try:
            if self.apply_pca:
                train_dataset, test_dataset, X_train, X_test, y_train, y_test = preprocess_and_split(
                    patches,
                    valid_labels,
                    self.transformation_config.batch_size,
                    ae=False
                )
            else:
                train_dataset, test_dataset, X_train, X_test, y_train, y_test = preprocess_and_split(
                    patches,
                    valid_labels,
                    self.transformation_config.batch_size,
                    ae=True
                )
            logging.info("Datasets prepared for training and testing")
            logging.info(f"Training Dataset: {train_dataset.element_spec}")
            logging.info(f"Testing Dataset: {test_dataset.element_spec}")

            return train_dataset, test_dataset, X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error during data preprocessing and splitting: {e}")
            raise CustomException(e, sys)

    def save_transformer(self, transformer_obj):
        """Save the transformer object for future use."""
        try:
            base_path = self.transformation_config.preprocessor_file_path
            suffix = "_cnn.pkl" if self.apply_pca else "_ae.pkl"
            path = base_path.replace(".pkl", suffix)
            save_object(path, transformer_obj)
            logging.info(f"Transformer object saved at {path}.")
        except Exception as e:
            logging.error(f"Error occurred while saving transformer object: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def load_transformer(file_path):
        """Load a previously saved transformer object."""
        try:
            return load_transformer(file_path)
        except Exception as e:
            logging.error(f"Error occurred while loading transformer object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, images, labels=None):
        """
        Initiates the complete data transformation process.
        """
        logging.info("Starting data transformation process")
        try:
            transformer = self.get_transformer_object(images)
            self.save_transformer(transformer)

            transformed_patches, valid_labels = self.transform_data(images, labels)
            valid_labels = valid_labels - 1  # Shift labels to 0-based index
            logging.info(f"Data transformed: {transformed_patches.shape} patches generated")

            logging.info("Preparing datasets for training and testing")
            train_dataset, test_dataset, X_train, X_test, y_train, y_test = self.prepare_datasets(transformed_patches, valid_labels)
            logging.info("Datasets prepared for training and testing")

            return train_dataset, test_dataset, X_train, X_test, y_train, y_test, transformer
        except Exception as e:
            logging.error(f"Error in data transformation process: {e}")
            raise CustomException(e, sys)


# Example usage:
if __name__ == "__main__":
    try:
        # Load full configuration
        CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))
        config = load_yaml(CONFIG_FILE)
        DATA_DIR = config['data_dir']
        DATASET_NAME = "Botswana"  # Example dataset name

        # --- Data Ingestion ---
        logging.info(f"Starting Data Ingestion for dataset: {DATASET_NAME} from {DATA_DIR}")
        data_ingestion = DataIngestion(data_dir=DATA_DIR, config_file=CONFIG_FILE)
        images, labels = data_ingestion.initiate_data_ingestion(dataset_name=DATASET_NAME)
        if images is None or labels is None:
            raise ValueError("Data ingestion failed, returned None.")
        logging.info(f"Data Ingestion completed. Image shape: {images.shape}, Labels shape: {labels.shape}")

        transformation_config=DataTransformationConfig(
            artifacts_root=config['transformation']['artifacts_root'],
            dataset_name=DATASET_NAME,
            data_dir=DATA_DIR,
            preprocessor_file_path=config['transformation']['preprocessor_file_path'],
            # Default fields after
            n_components=config['transformation']['pca_components'],  # Default PCA components
            patch_size=config['transformation']['patch_size'],     # Default patch size
            test_size=config['transformation']['test_size'],  # Fraction of data to use for testing
            batch_size=config['transformation']['batch_size']  # Batch size for TensorFlow datasets
        )

        # --- Data Transformation ---
        logging.info("Starting Data Transformation.")
        data_transformation = DataTransformation(config=transformation_config)
        train_dataset, test_dataset, X_train, X_test, y_train, y_test, transformer = data_transformation.initiate_data_transformation(images, labels)

        logging.info("Data transformation pipeline completed successfully.")
        logging.info(f"Training Dataset:{train_dataset.element_spec}")
        logging.info(f"Testing Dataset:{test_dataset.element_spec}")
        logging.info(f"Transformer Object:{transformer}")
        logging.info(f"X_train:{X_train.shape}")
        logging.info(f"X_test:{X_test.shape}")
        logging.info(f"y_train:{y_train.shape}")
        logging.info(f"y_test:{y_test.shape}")
        print(" Data transformation pipeline completed successfully.")

    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}") 
      