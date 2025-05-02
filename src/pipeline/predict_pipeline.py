import os
import sys
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.exception import CustomException
from src.logger import logging
from src.utils import load_yaml, extract_patches, load_transformer
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from tensorflow.keras.models import load_model


class PredictionPipeline:
    def __init__(self, config_path):
        try:
            self.config = load_yaml(config_path)
            self.data_dir = self.config['data_dir']
            self.model_path = self.config['model_trainer']['model_file_path']
            self.transformer_path = self.config['transformation']['preprocessor_file_path']
            logging.info("PredictionPipeline initialized with configuration from: %s", config_path)
        except Exception as e:
            logging.error("Error initializing PredictionPipeline.")
            raise CustomException(e, sys)

    def predict(self, dataset_name, image_data):
        try:
            logging.info(f"Starting prediction for dataset: {dataset_name}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at path: {self.model_path}")

            model = load_model(self.model_path)
            logging.info("Model loaded successfully.")

            transformer = DataTransformation.load_transformer(self.transformer_path)
            logging.info("Transformer object loaded successfully.")

            if not isinstance(image_data, np.ndarray):
                raise ValueError("Input image data must be a numpy array.")
            if image_data.ndim not in (2, 3):
                raise ValueError(f"Input image data must be 2D or 3D. Got shape: {image_data.shape}")

            if hasattr(transformer.pca, "components_"):
                transformed_data = transformer.transform_input_data(image_data)
            else:
                raise ValueError("PCA object is not properly fitted in the transformer.")

            patch_size = transformer.patch_size
            patches, _ = extract_patches(transformed_data, labels=None, patch_size=patch_size)

            if patches.ndim == 3:
                patches = np.expand_dims(patches, axis=-1)
            elif patches.ndim == 4:
                pass
            else:
                raise ValueError(f"Unexpected patch shape: {patches.shape}")

            predictions = model.predict(patches)
            logging.info(f"Raw model predictions shape: {predictions.shape}")
            logging.debug(f"Sample raw predictions:\n{predictions[:5]}")

            dataset_config = next((ds for ds in self.config['datasets'] if ds['name'] == dataset_name), None)
            if not dataset_config:
                raise ValueError(f"Dataset configuration for '{dataset_name}' not found in config file.")
            label_values = dataset_config['label_values']

            decoded_predictions = [label_values[np.argmax(pred)] for pred in predictions]
            prediction_indices = [np.argmax(pred) for pred in predictions]

            logging.info("Prediction completed successfully.")
            logging.info(f"Total predictions made: {len(decoded_predictions)}")
            logging.info("Sample predictions:")
            for i, pred in enumerate(decoded_predictions[:5]):
                logging.info(f"Prediction {i + 1}: {pred}")

            return decoded_predictions, prediction_indices

        except Exception as e:
            logging.error("Error during prediction.", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))
        DATASET_NAME = "Botswana"

        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"Config file not found at path: {CONFIG_FILE}")

        logging.info("Initializing PredictionPipeline with config file: %s", CONFIG_FILE)

        input_image_data = loadmat("DATA/Botswana/Botswana.mat")
        image_data = input_image_data.get('Botswana')

        if image_data is None:
            raise ValueError("The required image data is not found in the .mat file. Check the key name or file structure.")

        logging.info("Loaded image data shape: %s", image_data.shape)

        if image_data.ndim not in (2, 3):
            raise ValueError(f"Input image data must be 2D or 3D. Got shape: {image_data.shape}")

        # Load ground truth labels
        gt_data = loadmat("DATA/Botswana/Botswana_gt.mat")
        ground_truth = gt_data.get('Botswana_gt')

        if ground_truth is None:
            raise ValueError("Ground truth labels not found in the .mat file. Check the key name.")

        if ground_truth.shape != image_data.shape[:2]:
            raise ValueError("Shape mismatch between image data and ground truth labels.")

        # Initialize pipeline
        prediction_pipeline = PredictionPipeline(CONFIG_FILE)

        # Predict
        decoded_preds, pred_indices = prediction_pipeline.predict(dataset_name=DATASET_NAME, image_data=image_data)

        # Get patch size from config
        patch_size = prediction_pipeline.config['transformation']['patch_size']

        # Extract ground truth patches
        input_patches, label_patches = extract_patches(image_data, labels=ground_truth, patch_size=patch_size)
        labels_flat=label_patches.flatten()
        
        transformer=load_transformer("artifacts/preprocessor.pkl")

        # Optional: remove ignored or background class (e.g., 0)
        valid_mask = labels_flat > 0
        true_labels = labels_flat[valid_mask]
        input_patches=input_patches[valid_mask]

        transformed_data=transformer.transform_input_data(input_patches)
        
        if transformed_data.ndim==3:
            transformed_data=np.expand_dims(transformed_data, axis=-1)
            
        predictions=model.pre
        

        # Evaluation
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        report = classification_report(true_labels, pred_labels)

        logging.info("\n=== FINAL EVALUATION RESULTS ===")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info("\nClassification Report:\n%s", report)

        print("\nEvaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(report)

    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
    except FileNotFoundError as fnfe:
        logging.error(f"FileNotFoundError: {fnfe}")
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")











