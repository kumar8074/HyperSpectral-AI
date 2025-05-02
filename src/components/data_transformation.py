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

# Use absolute imports again
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
        Transform input data using PCA (if available).
        Patch extraction logic for prediction needs separate handling if required.

        Args:
            input_data (np.ndarray): The data to transform (e.g., H, W, C or patches).

        Returns:
            np.ndarray: Transformed data.
        """
        try:
            # --- Check if PCA transformation should be applied ---
            if self.pca is None:
                logging.info("Skipping PCA transformation as no PCA object is available (AE workflow).")
                # For AE workflow, prediction typically uses raw patches.
                # Assuming input_data is already in the correct format (e.g., patches).
                return input_data # Return data as-is

            logging.info("Applying PCA transformation to input data (CNN workflow).")
            # --- Apply PCA ---
            # Determine original shape and reshape if necessary
            original_shape = input_data.shape
            if input_data.ndim == 3: # (H, W, C) or (N, PatchH, PatchW, C)
                reshaped_data = input_data.reshape(-1, original_shape[-1])
            elif input_data.ndim == 2: # (N, C)
                reshaped_data = input_data
            else:
                raise ValueError("Input data must be 2D or 3D for PCA transformation")

            # Apply PCA transform
            transformed_data = self.pca.transform(reshaped_data)
            logging.info("Input data transformed successfully using PCA.")

            # Reshape back if original was 3D
            if input_data.ndim == 3:
                # Ensure the new shape matches (H, W, n_components) or (N, PatchH, PatchW, n_components)
                transformed_data = transformed_data.reshape(*original_shape[:-1], -1)
            
            # Note: This method currently ONLY applies PCA. If patch extraction 
            # is also needed at prediction time AFTER PCA, it should be handled 
            # separately or added here.

            return transformed_data # Return PCA-transformed data

        except Exception as e:
            logging.error(f"Error during input data transformation: {e}")
            raise CustomException(e, sys)

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
    batch_size: int = 32  # Batch size for TensorFlow datasets

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.transformation_config = config
        # Load the full config to check the model type
        config_path = os.path.join(project_root, 'config', 'config.yaml') # Assuming project_root is defined above
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

        Args:
            images (numpy.ndarray): The flattened training data to fit the PCA.

        Returns:
            Transformer: An object encapsulating PCA and patch size.
        """
        try:
            # Decide whether to fit PCA based on the model type
            if not self.apply_pca:
                logging.info("Skipping PCA fitting as per model type (AutoEncoder).")
                transformer = Transformer(pca=None, patch_size=self.transformation_config.patch_size)
                return transformer
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

        Args:
            images (numpy.ndarray): The input hyperspectral images.
            labels (numpy.ndarray, optional): The corresponding labels. Defaults to None.

        Returns:
            tuple: Transformed patches and valid labels (or None if labels are not provided).
        """
        try:
            if self.apply_pca:
                reduced_images = apply_pca(images, self.transformation_config.n_components)
            else:
                reduced_images = images

            # Extract patches (with or without labels based on availability)
            patches, valid_labels = extract_patches(
                reduced_images, labels, self.transformation_config.patch_size
            )

            # Normalize the patches (for both workflows)
            if self.apply_pca:
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
            # Determine the dataset format based on whether PCA was applied (linked to model type)
            ae_format = not self.apply_pca # If PCA wasn't applied, it's AE format
            if ae_format:
                logging.info("Creating TF datasets for AutoEncoder workflow.")
            else:
                logging.info("Creating TF datasets for CNN/other workflow.")

            train_dataset, test_dataset, X_train, X_test, y_train, y_test = preprocess_and_split(
                patches, valid_labels, self.transformation_config.test_size, self.transformation_config.batch_size,
                ae=ae_format # Pass the determined format
            )
            logging.info("Datasets prepared for training and testing")
            logging.info(f"Training Dataset:{train_dataset.element_spec}")
            logging.info(f"Testing Dataset:{test_dataset.element_spec}")
            return train_dataset, test_dataset, X_train, X_test, y_train, y_test
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
            base_path=self.transformation_config.preprocessor_file_path
            if self.apply_pca:
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
            valid_labels = valid_labels - 1 # Shift labels to 0-based index
            logging.info(f"Data transformed: {transformed_patches.shape} patches generated")

            # Prepare train and test datasets
            train_dataset, test_dataset, X_train, X_test, y_train, y_test = self.prepare_datasets(transformed_patches, valid_labels)
            logging.info("Datasets prepared for training and testing")

            return train_dataset, test_dataset, transformer, X_train, X_test, y_train, y_test
        
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
        train_dataset, test_dataset, transformer, X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(images, labels)

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
        logging.exception(f"An unexpected error occurred: {e}") # Use logging.exception for traceback