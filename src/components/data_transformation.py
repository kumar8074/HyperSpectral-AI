# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/components/data_transformation.py
# Description: This module handles the data transformation process for hyperspectral images.
#              It includes dimensionality reduction, patch extraction and splitting into train and test sets for specified HSI data.
# Author: LALAN KUMAR
# Created: [07-01-2025]
# Updated: [14-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)


from src.components.data_ingestion import DataIngestion
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Logging setup
from src.utils import (
    load_yaml, apply_pca, extract_patches,
    preprocess_and_split, save_object, load_transformer,
    normalize_patches
)  # Reusable utilities

class Transformer:
    """
    A class to encapsulate the PCA and patch size for transforming input data.
    """
    def __init__(self, pca, patch_size):
        self.pca = pca
        self.patch_size = patch_size

    def transform_input_data(self, input_data):
        """
        Transform input data using PCA and optionally extract patches if needed.

        Args:
            input_data: The data to transform.

        Returns:
            Transformed data.
        """
        try:
            if input_data.ndim == 3:
                original_shape = input_data.shape
                reshaped_data = input_data.reshape(-1, original_shape[-1])
            elif input_data.ndim == 2:
                reshaped_data = input_data
            else:
                raise ValueError("Input data must be 2D or 3D")

            transformed_data = self.pca.transform(reshaped_data)
            logging.info("Input data transformed successfully using PCA.")

            if input_data.ndim == 3:
                transformed_data = transformed_data.reshape(original_shape[0], original_shape[1], -1)

            return transformed_data
        except Exception as e:
            logging.error(f"Error during input data transformation: {e}")
            raise CustomException(e, sys)

@dataclass
class DataTransformationConfig:
    """
    Configuration class for defining parameters used during data transformation.
    """
    pca_components: int  # Number of principal components to retain
    patch_size: int  # Size of patches to extract (e.g., 7x7)
    test_size: float  # Fraction of data to use for testing
    batch_size: int  # Batch size for TensorFlow datasets
    transformer_obj_file_path: str  # Path to save the transformer object
    use_pca: bool  # Whether to apply PCA (True for CNN, False for AE)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.transformation_config = config

    def get_transformer_object(self, images):
        """
        Create a data transformer object containing PCA and patch extraction configurations.

        Args:
            images (numpy.ndarray): The flattened training data to fit the PCA.

        Returns:
            Transformer: An object encapsulating PCA and patch size.
        """
        try:
            if not self.transformation_config.use_pca:
                logging.info("Skipping PCA as per configuration (AutoEncoder workflow).")
                transformer=Transformer(pca=None, patch_size=self.transformation_config.patch_size)
                return transformer

            pca = PCA(n_components=self.transformation_config.pca_components)
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

        Args:
            images (numpy.ndarray): The input hyperspectral images.
            labels (numpy.ndarray, optional): The corresponding labels. Defaults to None.

        Returns:
            tuple: Transformed patches and valid labels (or None if labels are not provided).
        """
        try:
            if self.transformation_config.use_pca:
                reduced_images = apply_pca(images, self.transformation_config.pca_components)
            else:
                reduced_images = images

            # Extract patches (with or without labels based on availability)
            patches, valid_labels = extract_patches(
                reduced_images, labels, self.transformation_config.patch_size
            )

            # Normalize the patches (for both workflows)
            if self.transformation_config.use_pca:
                normalized_patches = normalize_patches(patches,method='pca_output') # For CNN
            else:
                normalized_patches = normalize_patches(patches, method='per_band') # For AE

            return normalized_patches, valid_labels
        except Exception as e:
            logging.error(f"Error during data transformation: {e}")
            raise CustomException(e, sys)

    def prepare_datasets(self, patches, valid_labels):
        """
        Preprocess the transformed data and split into training and testing datasets.

        Args:
            patches (numpy.ndarray): Extracted data patches.
            valid_labels (numpy.ndarray): Corresponding labels for the patches.

        Returns:
            tuple: Training and testing datasets.
        """
        try:
            if not self.transformation_config.use_pca:
                logging.info("Creating TF datasets for AutoEncoder workflow.")
                train_dataset, test_dataset = preprocess_and_split(
                    patches, valid_labels, self.transformation_config.test_size, self.transformation_config.batch_size,
                    ae=True
                )
                logging.info(f"Training Dataset:{train_dataset.element_spec}")
                logging.info(f"Testing Dataset:{test_dataset.element_spec}")
            else:
                logging.info("Creating TF datasets for CNN workflow.")
                train_dataset, test_dataset = preprocess_and_split(
                    patches, valid_labels, self.transformation_config.test_size, self.transformation_config.batch_size,
                    ae=False
                )
                logging.info(f"Training Dataset:{train_dataset.element_spec}")
                logging.info(f"Testing Dataset:{test_dataset.element_spec}")
                
            return train_dataset, test_dataset
        except Exception as e:
            logging.error(f"Error during data preprocessing and splitting: {e}")
            raise CustomException(e, sys)

    def save_transformer(self, transformer_obj):
        """
        Save the transformer object for future use.

        Args:
            transformer_obj (Transformer): The transformer object to save.
        """
        try:
            base_path=self.transformation_config.transformer_obj_file_path
            if self.transformation_config.use_pca:
                path=base_path.replace(".pkl","_cnn.pkl")
            else:
                path=base_path.replace(".pkl","_ae.pkl")
                
            save_object(path, transformer_obj)
            logging.info(f"Transformer object saved at {path}.")
        except Exception as e:
            logging.error(f"Error occurred while saving transformer object: {e}")
            raise CustomException(e, sys)

    @staticmethod
    def load_transformer(file_path):
        """
        Load a previously saved transformer object.

        Args:
            file_path (str): Path to the saved transformer object.

        Returns:
            Transformer: Loaded transformer object.
        """
        try:
            return load_transformer(file_path)
        except Exception as e:
            logging.error(f"Error occurred while loading transformer object: {e}")
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self, images, labels=None):
        """
        Initiates the complete data transformation process by creating the transformer,
        applying transformations, and preparing datasets.

        Args:
            images (numpy.ndarray): The input hyperspectral images.
            labels (numpy.ndarray, optional): The corresponding labels. Defaults to None.

        Returns:
            tuple: A tuple containing (train_dataset, test_dataset, transformer_object)
        """
        logging.info("Starting data transformation process")
        try:
            # Create and save transformer object
            transformer = self.get_transformer_object(images)
            self.save_transformer(transformer)
            logging.info("Transformer object created and saved successfully")

            # Transform data
            transformed_patches, valid_labels = self.transform_data(images, labels)
            logging.info(f"Data transformed: {transformed_patches.shape} patches generated")

            # Prepare train and test datasets
            train_dataset, test_dataset = self.prepare_datasets(transformed_patches, valid_labels)
            logging.info("Datasets prepared for training and testing")

            return train_dataset, test_dataset, transformer
        
        except Exception as e:
            logging.error(f"Error in data transformation process: {e}")
            raise CustomException(e, sys)


# Example usage:
if __name__ == "__main__":
    try:
        # Paths and configurations
        CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))
        config = load_yaml(CONFIG_FILE)
        DATA_DIR = config['data_dir']
        DATASET_NAME = "Botswana"  # Example dataset name
        USE_PCA = config['transformation'].get('use_pca', True)  # Explicitly control workflow

        # Data ingestion
        data_ingestion = DataIngestion(data_dir=DATA_DIR, config_file=CONFIG_FILE)
        images, labels = data_ingestion.initiate_data_ingestion(dataset_name=DATASET_NAME)

        # Data transformation configurations
        transformation_config = DataTransformationConfig(
            pca_components=config['transformation']['pca_components'],
            patch_size=config['transformation']['patch_size'],
            test_size=config['transformation']['test_size'],
            batch_size=config['transformation']['batch_size'],
            transformer_obj_file_path=config['transformation']['preprocessor_file_path'],
            use_pca=USE_PCA
        )

        # Data transformation
        data_transformation = DataTransformation(config=transformation_config)
        
        # Use the new initiate_data_transformation method
        train_dataset, test_dataset, transformer = data_transformation.initiate_data_transformation(images, labels)

        logging.info("Data transformation completed successfully.")
        print("✅ Data transformation pipeline completed successfully.")

    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")