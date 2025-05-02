# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/components/data_ingestion.py
# Description: This file contains the data ingestion component of the hyperspectral image classification project.
# Author: LALAN KUMAR
# Created: [07-01-2025]
# Updated: [02-05-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import os
import sys
import argparse
from dataclasses import dataclass

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import load_yaml, load_hyperspectral_data  # Import reusable utilities
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Logging setup


@dataclass
class DataIngestionConfig:
    """
    Configuration class for defining paths used during data ingestion.
    """
    data_dir: str  # Root directory containing datasets
    config_file: str  # Path to the configuration YAML file

class DataIngestion:
    def __init__(self, data_dir: str, config_file: str):
        self.ingestion_config = DataIngestionConfig(
            data_dir=data_dir,
            config_file=config_file
        )

    def initiate_data_ingestion(self, dataset_name: str):
        """
        Initiates the data ingestion process by loading the specified dataset.

        Args:
            dataset_name (str): The name of the dataset to load.

        Returns:
            tuple: A tuple containing the loaded images and labels.
        """
        logging.info("Starting data ingestion for dataset: %s", dataset_name)
        try:
            # Load configuration from the YAML file
            logging.info("Loading configuration from: %s", self.ingestion_config.config_file)
            config = load_yaml(self.ingestion_config.config_file)

            # Load the specified dataset
            logging.info("Loading dataset: %s", dataset_name)
            images, labels = load_hyperspectral_data(
                data_dir=self.ingestion_config.data_dir,
                dataset_name=dataset_name,
                config=config
            )

            logging.info("Successfully loaded dataset: %s", dataset_name)
            return images, labels

        except FileNotFoundError as fnfe:
            logging.error("File not found: %s", fnfe)
            raise CustomException(fnfe, sys)
        except Exception as e:
            logging.error("Error during data ingestion for dataset: %s", dataset_name)
            raise CustomException(e, sys)

# Example usage:
if __name__ == "__main__":
    try:
        # Define project root (assuming the existing calculation is correct)
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))

        # --- Argument Parsing --- 
        parser = argparse.ArgumentParser(description="Run data ingestion for a hyperspectral dataset.")
        parser.add_argument("--data_dir", type=str, default="DATA", 
                            help="Root directory containing datasets (default: DATA)")
        parser.add_argument("--dataset_name", type=str, default="Botswana", 
                            help="Name of the dataset to ingest (e.g., KSC, PaviaU) (default: Botswana)")
        parser.add_argument("--config_file", type=str, 
                            default=os.path.join(project_root, 'config', 'config.yaml'),
                            help="Path to the configuration YAML file.")
        
        args = parser.parse_args()

        # --- Use parsed arguments --- 
        DATA_DIR = args.data_dir
        CONFIG_FILE = args.config_file
        DATASET_NAME = args.dataset_name

        # Ensure config file path exists
        if not os.path.exists(CONFIG_FILE):
            logging.error(f"Configuration file not found at: {CONFIG_FILE}")
            raise FileNotFoundError(f"Configuration file not found at: {CONFIG_FILE}")

        logging.info(f"Using Data Directory: {DATA_DIR}")
        logging.info(f"Using Config File: {CONFIG_FILE}")
        logging.info(f"Using Dataset Name: {DATASET_NAME}")

        # Initialize DataIngestion
        data_ingestion = DataIngestion(data_dir=DATA_DIR, config_file=CONFIG_FILE)

        # Ingest data
        images, labels = data_ingestion.initiate_data_ingestion(dataset_name=DATASET_NAME)

        logging.info("Data ingestion completed successfully.")
        logging.info(f"Image shape: {images.shape}, Label shape: {labels.shape}")

    except FileNotFoundError as fnfe:
        logging.error(f"File not found error during setup: {fnfe}") 
    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")










