import os
import sys

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
    
from src.exception import CustomException  # Custom exception class
from src.logger import logging  # Logging setup
from src.utils import load_yaml, evaluate_model  # Reusable utilities
from src.components.data_ingestion import DataIngestion  # Data ingestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig  # Data transformation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig  # Model training

def train_pipeline():
    try:
        # Paths and configurations
        CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))
        config = load_yaml(CONFIG_FILE)
        DATA_DIR = config['data_dir']
        DATASET_NAME = config['model_trainer']['dataset']  # Dynamically get the dataset name from the config

        # Data ingestion
        logging.info(f"Starting data ingestion for dataset: {DATASET_NAME}")
        data_ingestion = DataIngestion(data_dir=DATA_DIR, config_file=CONFIG_FILE)
        images, labels = data_ingestion.initiate_data_ingestion(dataset_name=DATASET_NAME)
        logging.info(f"Successfully loaded dataset: {DATASET_NAME}")

        # Data transformation configurations
        transformation_config = DataTransformationConfig(
            pca_components=config['transformation']['pca_components'],
            patch_size=config['transformation']['patch_size'],
            test_size=config['transformation']['test_size'],
            batch_size=config['transformation']['batch_size'],
            transformer_obj_file_path=config['transformation']['preprocessor_file_path']
        )

        # Data transformation
        data_transformation = DataTransformation(config=transformation_config)
        patches, valid_labels = data_transformation.transform_data(images, labels)
        train_dataset, test_dataset = data_transformation.preprocess_and_split(patches, valid_labels)

        # Save the transformer object
        transformer = data_transformation.get_transformer_object(images)
        data_transformation.save_transformer(transformer)

        # Get the number of classes and input channels
        dataset_config = next(ds for ds in config['datasets'] if ds['name'] == DATASET_NAME)
        n_classes = len(dataset_config['label_values'])
        in_channels = config['transformation']['pca_components']

        # Model trainer configurations
        trainer_config = ModelTrainerConfig(
            model_name=config['model_trainer']['model'],
            epochs=config['model_trainer']['epochs'],
            learning_rate=config['model_trainer']['learning_rate'],
            loss=config['model_trainer']['loss'],
            optimizer=config['model_trainer']['optimizer'],
            metrics=config['model_trainer']['metrics'],
            model_save_path=config['model_trainer']['model_file_path']
        )

        # Model training
        model_trainer = ModelTrainer(config=trainer_config)
        model = model_trainer.train(train_dataset, test_dataset, n_classes, in_channels)

        logging.info("Model training completed successfully.")

        # Evaluate the model
        logging.info("Starting model evaluation.")
        X_test, y_test = next(iter(test_dataset))  # Extract test data from the dataset
        evaluate_model(model, X_test, y_test, dataset_config['label_values'])

    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    train_pipeline()
    
    
