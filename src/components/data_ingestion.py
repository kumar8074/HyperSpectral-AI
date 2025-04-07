import os
import sys
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
        # Define paths
        DATA_DIR = "DATA"  # Change as per your directory
        CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))

        # Initialize DataIngestion
        data_ingestion = DataIngestion(data_dir=DATA_DIR, config_file=CONFIG_FILE)

        # Specify dataset to load
        DATASET_NAME = "Botswana"  # Example dataset name

        # Ingest data
        images, labels = data_ingestion.initiate_data_ingestion(dataset_name=DATASET_NAME)

        logging.info("Data ingestion completed successfully.")
        logging.info(f"Image shape: {images.shape}, Label shape: {labels.shape}")

    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")





