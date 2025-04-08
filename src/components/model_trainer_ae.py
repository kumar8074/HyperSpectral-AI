import os
import sys
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Add project root to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.exception import CustomException
from src.logger import logging
from src.utils import load_yaml
from src.models.autoencoder_model import HyperspectralAE
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.data_ingestion import DataIngestion

@dataclass
class AEModelTrainerConfig:
    epochs: int
    learning_rate: float
    model_save_path: str
    batch_size: int
    patience: int  # Early stopping patience

class AEModelTrainer:
    def __init__(self, config: AEModelTrainerConfig):
        self.trainer_config = config

    def train(self, train_dataset, val_dataset, n_classes, in_channels):
        try:
            model = HyperspectralAE(in_channels=in_channels, n_classes=n_classes)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.trainer_config.learning_rate),
                loss={
                    "reconstruction": "mse",
                    "classification": "sparse_categorical_crossentropy"
                },
                metrics={
                    "reconstruction": "mse",
                    "classification": "accuracy"
                }
            )

            logging.info("AutoEncoder model compiled successfully.")

            # Ensure model directory exists
            model_dir = os.path.dirname(self.trainer_config.model_save_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            callbacks = [
                EarlyStopping(
                    monitor='val_classification_accuracy',
                    patience=self.trainer_config.patience,
                    restore_best_weights=True,
                    mode='max'
                ),
                ModelCheckpoint(
                    filepath=self.trainer_config.model_save_path,
                    monitor='val_classification_accuracy',
                    save_best_only=True
                )
            ]

            logging.info("Starting AutoEncoder model training.")

            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.trainer_config.epochs,
                callbacks=callbacks
            )

            logging.info("AutoEncoder model training completed.")
            return model

        except Exception as e:
            logging.error(f"Error during AE model training: {e}")
            raise CustomException(e, sys)


def prepare_ae_datasets(X_train, X_test, y_train, y_test, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, {"reconstruction": X_train, "classification": y_train})
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, {"reconstruction": X_test, "classification": y_test})
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


if __name__ == "__main__":
    try:
        CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config/config.yaml'))
        config = load_yaml(CONFIG_FILE)

        DATA_DIR = config['data_dir']
        DATASET_NAME = config['model_trainer']['dataset']

        # Data ingestion
        data_ingestion = DataIngestion(data_dir=DATA_DIR, config_file=CONFIG_FILE)
        images, labels = data_ingestion.initiate_data_ingestion(dataset_name=DATASET_NAME)

        dataset_index = next((i for i, d in enumerate(config['datasets']) if d['name'] == DATASET_NAME), -1)
        if dataset_index == -1:
            raise ValueError(f"Dataset {DATASET_NAME} not found in configuration")

        # Data transformation config (no PCA for AE)
        transformation_config = DataTransformationConfig(
            patch_size=config['transformation']['patch_size'],
            test_size=config['transformation']['test_size'],
            batch_size=config['transformation']['batch_size'],
            transformer_obj_file_path=config['transformation']['preprocessor_file_path'],
            use_pca=False,
            pca_components=None
        )

        data_transformation = DataTransformation(config=transformation_config)
        patches, valid_labels = data_transformation.transform_data(images, labels)

        batch_size = config['transformation']['batch_size']
        train_dataset, test_dataset = prepare_ae_datasets(
            patches['train'], patches['test'],
            valid_labels['train'], valid_labels['test'],
            batch_size
        )

        in_channels = images.shape[-1]
        n_classes = len(config['datasets'][dataset_index]['label_values'])

        ae_trainer_config = AEModelTrainerConfig(
            epochs=config['model_trainer']['epochs'],
            learning_rate=config['model_trainer']['learning_rate'],
            model_save_path=config['model_trainer']['model_file_path_ae'],
            batch_size=batch_size,
            patience=10
        )

        model_trainer = AEModelTrainer(config=ae_trainer_config)
        model = model_trainer.train(train_dataset, test_dataset, n_classes, in_channels)

        logging.info("AE model training script finished successfully.")

    except Exception as e:
        logging.error(f"Error in AE model training script: {e}")
        raise CustomException(e, sys)



