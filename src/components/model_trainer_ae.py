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
from src.utils import get_optimizer, get_loss, get_metric, load_yaml  # Reusable utilities
from src.models.autoencoder_model import HyperspectralAE  # AutoEncoder Model definition
from src.components.data_transformation import DataTransformation, DataTransformationConfig  # Data transformation
from src.components.data_ingestion import DataIngestion  # Data ingestion

@dataclass
class AEModelTrainerConfig:
    """
    Configuration class for defining parameters used during AutoEncoder model training.
    """
    epochs: int  # Number of epochs to train
    learning_rate: float  # Learning rate for the optimizer
    loss: str  # Loss function to use for classification
    optimizer: str  # Optimizer to use
    metrics: list  # Metrics to track during training
    model_save_path: str  # Path to save the trained model
    reconstruction_weight: float = 0.5  # Weight for reconstruction loss in the combined loss


class AEModelTrainer:
    def __init__(self, config: AEModelTrainerConfig):
        self.trainer_config = config
        
    def train(self, train_dataset, test_dataset, n_classes, in_channels):
        """
        Train the AutoEncoder model using the provided datasets and configuration.

        Args:
            train_dataset: TensorFlow dataset for training.
            test_dataset: TensorFlow dataset for testing.
            n_classes: Number of output classes.
            in_channels: Number of input channels.

        Returns:
            model: The trained TensorFlow Keras model.
        """
        try:
            # Validate dataset compatibility
            for data, label in train_dataset.take(1):
                input_shape = data.shape[1:]
                if input_shape[-1] != in_channels:
                    raise ValueError(f"Mismatch in input channels. Expected: {in_channels}, Got: {input_shape[-1]}")
                
            # Initialize the AutoEncoder model
            model = HyperspectralAE(in_channels=in_channels, n_classes=n_classes)
            logging.info("AutoEncoder model initialized with in_channels: %s, n_classes: %s", in_channels, n_classes)

            # Define optimizer
            optimizer = get_optimizer(self.trainer_config.optimizer)(learning_rate=self.trainer_config.learning_rate)
            
            # Define metrics for tracking
            metrics = {
                "classifier_output": [get_metric(metric) for metric in self.trainer_config.metrics]
            }
            
            # Define loss functions
            classification_loss = get_loss(self.trainer_config.loss)
            reconstruction_loss = tf.keras.losses.MeanSquaredError()
            
            # Ensure model save directory exists
            model_dir = os.path.dirname(self.trainer_config.model_save_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                logging.info(f"Model directory created at: {model_dir}")

            # Check for None values in the datasets
            if train_dataset is None or test_dataset is None:
                raise ValueError("Train or test dataset contains None values.")
            
            # Define custom training step
            @tf.function
            def train_step(x, y):
                with tf.GradientTape() as tape:
                    # Forward pass
                    decoded, classified = model(x, training=True)
                    
                    # Calculate losses
                    recon_loss = reconstruction_loss(x, decoded)
                    class_loss = classification_loss(y, classified)
                    
                    # Combined loss with weighting
                    total_loss = (1 - self.trainer_config.reconstruction_weight) * class_loss + \
                                 self.trainer_config.reconstruction_weight * recon_loss
                
                # Calculate gradients and update weights
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                return {
                    "total_loss": total_loss,
                    "reconstruction_loss": recon_loss,
                    "classification_loss": class_loss
                }
            
            # Define test step
            @tf.function
            def test_step(x, y):
                # Forward pass
                decoded, classified = model(x, training=False)
                
                # Calculate losses
                recon_loss = reconstruction_loss(x, decoded)
                class_loss = classification_loss(y, classified)
                
                # Combined loss with weighting
                total_loss = (1 - self.trainer_config.reconstruction_weight) * class_loss + \
                             self.trainer_config.reconstruction_weight * recon_loss
                
                # Update metrics
                for m in metrics["classifier_output"]:
                    m.update_state(y, classified)
                
                return {
                    "val_total_loss": total_loss,
                    "val_reconstruction_loss": recon_loss, 
                    "val_classification_loss": class_loss
                }
                
            # Training loop
            logging.info("Starting AutoEncoder model training")
            best_val_loss = float('inf')
            best_epoch = 0
            
            for epoch in range(self.trainer_config.epochs):
                # Training
                train_losses = {
                    "total_loss": [],
                    "reconstruction_loss": [],
                    "classification_loss": []
                }
                
                for x_batch, y_batch in train_dataset:
                    batch_losses = train_step(x_batch, y_batch)
                    for k, v in batch_losses.items():
                        train_losses[k].append(v.numpy())
                
                # Calculate average training losses
                avg_train_losses = {k: sum(v) / len(v) for k, v in train_losses.items()}
                
                # Validation
                test_losses = {
                    "val_total_loss": [],
                    "val_reconstruction_loss": [],
                    "val_classification_loss": []
                }
                
                for x_batch, y_batch in test_dataset:
                    batch_losses = test_step(x_batch, y_batch)
                    for k, v in batch_losses.items():
                        test_losses[k].append(v.numpy())
                
                # Calculate average validation losses
                avg_test_losses = {k: sum(v) / len(v) for k, v in test_losses.items()}
                
                # Calculate metric values
                metric_values = {m.name: m.result().numpy() for m in metrics["classifier_output"]}
                
                # Log progress
                log_msg = f"Epoch {epoch+1}/{self.trainer_config.epochs} - "
                log_msg += " - ".join([f"{k}: {v:.4f}" for k, v in {**avg_train_losses, **avg_test_losses, **metric_values}.items()])
                logging.info(log_msg)
                
                # Reset metrics for next epoch
                for m in metrics["classifier_output"]:
                    m.reset_states()
                
                # Save best model
                if avg_test_losses["val_total_loss"] < best_val_loss:
                    best_val_loss = avg_test_losses["val_total_loss"]
                    best_epoch = epoch + 1
                    model.save_weights(self.trainer_config.model_save_path)
                    logging.info(f"Model saved at epoch {best_epoch} with val_total_loss: {best_val_loss:.4f}")
            
            logging.info(f"AutoEncoder training completed. Best model saved at epoch {best_epoch} with val_total_loss: {best_val_loss:.4f}")
            
            # Load best weights
            model.load_weights(self.trainer_config.model_save_path)
            
            return model

        except Exception as e:
            logging.error("Error during AutoEncoder model training: %s", str(e))
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

        # Data transformation configurations - AutoEncoder doesn't use PCA
        transformation_config = DataTransformationConfig(
            pca_components=config['transformation']['pca_components'],
            patch_size=config['transformation']['patch_size'],
            test_size=config['transformation']['test_size'],
            batch_size=config['transformation']['batch_size'],
            transformer_obj_file_path=config['transformation']['preprocessor_file_path'],
            use_pca=False  # AutoEncoder performs its own dimensionality reduction
        )

        # Data transformation - For AutoEncoder, we extract patches but skip PCA
        data_transformation = DataTransformation(config=transformation_config)
        patches, valid_labels = data_transformation.transform_data(images, labels)
        train_dataset, test_dataset = data_transformation.prepare_datasets(patches, valid_labels)

        # Get the number of classes and input channels
        n_classes = len(config['datasets'][dataset_index]['label_values'])
        in_channels = images.shape[-1]  # Original number of spectral bands

        # AutoEncoder model trainer configurations
        trainer_config = AEModelTrainerConfig(
            epochs=config['model_trainer']['epochs'],
            learning_rate=config['model_trainer']['learning_rate'],
            loss=config['model_trainer']['loss'],
            optimizer=config['model_trainer']['optimizer'],
            metrics=config['model_trainer']['metrics'],
            model_save_path=config['model_trainer']['model_file_path_ae'],
            reconstruction_weight=0.5  # You might want to add this to config.yaml
        )

        # Model training
        model_trainer = AEModelTrainer(config=trainer_config)
        model = model_trainer.train(train_dataset, test_dataset, n_classes, in_channels)

        logging.info("AutoEncoder model training completed successfully.")
    except Exception as e:
        logging.error(f"Error in AutoEncoder model training script: {e}")
        raise CustomException(e, sys)
