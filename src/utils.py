# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/utils.py
# Description: This file contains the various utility functions used in the hyperspectral image classification project.
# Author: LALAN KUMAR
# Created: [08-01-2025]
# Updated: [02-05-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

"""Utility functions for the Hyperspectral Image Classification project.

Includes functions for data loading, transformation, preprocessing, model saving/loading,
and plotting.
"""

import os
import pickle
import sys
import logging
import yaml
import scipy.io as sio
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from typing import Optional

from .logger import logging as project_logger
from .exception import CustomException


################################ USED IN DATA INGESTION ################################

# Load a YAML file from the given file path
def load_yaml(file_path):
    """
    Load a YAML file from the given file path.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: The data loaded from the YAML file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found at: {file_path}")

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    return data

def load_hyperspectral_data(data_dir, dataset_name, config):
    """
    Load hyperspectral data and labels from .mat files, with dynamic key extraction and robust logging.
    Args:
        data_dir (str): Directory containing the .mat files.
        dataset_name (str): The name of the dataset (e.g., 'Botswana', 'PaviaC', etc.).
        config (dict): The configuration dictionary from the config.yaml file.
    Returns:
        images (ndarray): Hyperspectral image data.
        labels (ndarray): Ground-truth labels.
    """
    try:
        # Check if dataset exists in config
        dataset_config = next((dataset for dataset in config['datasets'] if dataset['name'] == dataset_name), None)
        if not dataset_config:
            project_logger.error(f"Dataset '{dataset_name}' not found in the config.")
            raise ValueError(f"Dataset '{dataset_name}' not found in the config.")
        
        # Build the dataset path and verify it exists
        dataset_path = os.path.join(data_dir, dataset_config['name'])
        if not os.path.exists(dataset_path):
            project_logger.error(f"Dataset folder '{dataset_name}' not found in '{dataset_path}'.")
            raise FileNotFoundError(f"Dataset folder '{dataset_name}' not found in '{dataset_path}'.")
        
        project_logger.info(f"Searching for .mat files in: {dataset_path}")
        
        # Find image and label files
        image_file = next((f for f in os.listdir(dataset_path) if f.endswith('.mat') and '_gt' not in f), None)
        label_file = next((f for f in os.listdir(dataset_path) if f.endswith('_gt.mat')), None)
        
        if not image_file or not label_file:
            project_logger.error(f"Image or label .mat files not found in the '{dataset_name}' dataset directory.")
            raise FileNotFoundError(f"Image or label .mat files not found in the '{dataset_name}' dataset directory.")
        
        project_logger.info(f"Found image file: {image_file}")
        project_logger.info(f"Found label file: {label_file}")
        
        # Load image data
        image_data = sio.loadmat(os.path.join(dataset_path, image_file))
        project_logger.debug(f"Keys in the image file '{image_file}': {image_data.keys()}")
        
        # Dynamic key extraction for image
        image_key = next((key for key in image_data.keys() if dataset_name.lower() in key.lower() or 
                          'data' in key.lower() or key.lower() in ['pavia', 'ksc', 'botswana']), None)
        
        if image_key is None:
            project_logger.error(f"Image data key for '{dataset_name}' not found in the image file {image_file}")
            raise KeyError(f"Image data key for '{dataset_name}' not found in the image file {image_file}.")
        
        project_logger.info(f"Found image key: {image_key}")
        images = image_data.get(image_key)
        
        if images is None:
            project_logger.error(f"Image data not found in the key '{image_key}' in the file {image_file}")
            raise KeyError(f"Image data not found in the key '{image_key}' in the file {image_file}.")
        
        # Dynamic key extraction for label
        label_data = sio.loadmat(os.path.join(dataset_path, label_file))
        project_logger.debug(f"Keys in the label file '{label_file}': {label_data.keys()}")
        
        label_key = next((key for key in label_data.keys() if 'gt' in key.lower() or 'labels' in key.lower()), None)
        
        if label_key is None:
            project_logger.error(f"Label data key for '{dataset_name}' not found in the label file {label_file}")
            raise KeyError(f"Label data key for '{dataset_name}' not found in the label file {label_file}.")
        
        project_logger.info(f"Found label key: {label_key}")
        labels = label_data.get(label_key)
        
        if labels is None:
            project_logger.error(f"Label data not found in the key '{label_key}' in the file {label_file}")
            raise KeyError(f"Label data not found in the key '{label_key}' in the file {label_file}.")
        
        # Log information about the loaded data
        project_logger.info(f"Image shape: {images.shape}")
        project_logger.info(f"Label shape: {labels.shape}")
        project_logger.info(f"Unique labels: {set(labels.flatten())}")
        project_logger.info("Successfully loaded hyperspectral data and labels")
        
        return images, labels
        
    except Exception as e:
        project_logger.error(f"Error occurred while loading hyperspectral data: {str(e)}", exc_info=True)
        raise

################################ USED IN DATA INGESTION ############################################

################################ USED IN DATA TRANSFORMATION ########################################

# Applies PCA to the hyperspectral data (only used for CNN Based Approach)
def apply_pca(images, n_components):
    """
    Apply PCA to reduce the dimensionality of the hyperspectral data.
    Args:
        images (ndarray): Hyperspectral image data of shape (H, W, C).
        n_components (int): Number of principal components to retain.
    Returns:
        reduced_images (ndarray): PCA-reduced hyperspectral image data of shape (H, W, n_components).
    """
    try:
        project_logger.info(f"Starting PCA reduction with n_components={n_components}")
        h, w, c = images.shape
        project_logger.info(f"Input image shape: height={h}, width={w}, channels={c}")
        reshaped_images = images.reshape(-1, c)
        project_logger.info("Initializing PCA")
        pca = PCA(n_components=n_components)
        project_logger.info("Applying PCA transformation")
        reduced_data = pca.fit_transform(reshaped_images)
        reduced_images = reduced_data.reshape(h, w, n_components)
        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        project_logger.info(f"PCA completed: Original bands = {c}, Reduced bands = {n_components}")
        project_logger.info(f"Total explained variance: {explained_variance:.2f}%")
        return reduced_images
    except Exception as e:
        project_logger.error(f"Error in PCA application: {str(e)}", exc_info=True)
        raise

# Extracts patches from Hyperspectral data (Utilized both for CNN and AutoEncoder based Approaches.)
def extract_patches(
    images: np.ndarray,
    labels: Optional[np.ndarray] = None,
    patch_size: int = 7,
    normalize_per_patch: bool = True,
    standardize_patches: bool = False
):
    """
    Extract patches ONLY from labeled pixels (label != 0) without padding the input image,
    with optional per-patch normalization and standardization.
    Args:
        images (ndarray): Hyperspectral image data of shape (H, W, C).
        labels (ndarray, optional): Ground-truth labels of shape (H, W). Must be provided for labeled patches.
        patch_size (int): Size of the patch to extract (e.g., 7 for 7x7 patches).
        normalize_per_patch (bool): If True, scales each patch by its own max value to [0, 1].
        standardize_patches (bool): If True, after normalization, standardizes each patch to zero mean and unit variance.
    Returns:
        patches (ndarray): Extracted patches of shape (num_patches, patch_size, patch_size, C).
        valid_labels (ndarray or None): The corresponding non-zero labels for each patch.
        locations (ndarray): The (row, column) coordinates of the center pixel for each extracted patch in the original image.
    """
    patch_half = patch_size // 2
    h, w, _ = images.shape
    patches = []
    valid_labels = []
    locations = []

    # Cast once; per-patch scaling handled below
    images = images.astype(np.float32)

    for i in range(patch_half, h - patch_half):
        for j in range(patch_half, w - patch_half):
            if labels is not None and labels[i, j] == 0:
                continue

            patch = images[
                i - patch_half : i + patch_half + 1,
                j - patch_half : j + patch_half + 1,
                :
            ].copy()

            # Per-patch normalization
            if normalize_per_patch:
                max_val = np.max(patch)
                if max_val > 0:
                    patch = patch / max_val

            # Optional standardization (zero mean, unit variance)
            if standardize_patches:
                std = patch.std()
                if std > 0:
                    patch = (patch - patch.mean()) / std

            patches.append(patch)
            locations.append((i, j))
            if labels is not None:
                valid_labels.append(labels[i, j])

    patches_array = np.stack(patches, axis=0)
    locations_array = np.array(locations)
    valid_labels_array = np.array(valid_labels) if labels is not None else None

    return patches_array, valid_labels_array, locations_array

# Noramalizes the extracted patches
def normalize_patches(patches, method='pca_output'): # For AE use per_band.
    """
    Normalize hyperspectral image patches or PCA-reduced patches.

    Parameters:
        patches (np.ndarray): Input array of shape (num_patches, height, width, channels/components).
        method (str): One of 'per_band', 'per_patch', 'pca_output'.

    Returns:
        np.ndarray: Normalized patches of the same shape.
    """
    patches = patches.astype(np.float32)

    if np.isnan(patches).any():
        raise ValueError("Input contains NaNs before normalization.")

    if method == 'per_band':
        # Suitable for AE+Classifier (original spectral bands)
        num_patches, height, width, channels = patches.shape
        reshaped = patches.reshape(-1, channels)

        # Clip extreme values to improve robustness
        reshaped = np.clip(
            reshaped,
            np.percentile(reshaped, 1, axis=0),
            np.percentile(reshaped, 99, axis=0)
        )

        min_vals = reshaped.min(axis=0)
        max_vals = reshaped.max(axis=0)
        denom = max_vals - min_vals
        denom[denom == 0] = 1  # Avoid division by zero

        normalized = (reshaped - min_vals) / denom
        return normalized.reshape(num_patches, height, width, channels)

    elif method == 'per_patch':
        # Normalize each patch individually (can be noisy)
        min_vals = patches.min(axis=(1, 2, 3), keepdims=True)
        max_vals = patches.max(axis=(1, 2, 3), keepdims=True)
        denom = max_vals - min_vals
        denom[denom == 0] = 1

        return (patches - min_vals) / denom

    elif method == 'pca_output':
        # Suitable after PCA (for CNN workflow)
        num_patches, height, width, components = patches.shape
        reshaped = patches.reshape(-1, components)

        min_vals = reshaped.min(axis=0)
        max_vals = reshaped.max(axis=0)
        denom = max_vals - min_vals
        denom[denom == 0] = 1

        normalized = (reshaped - min_vals) / denom
        return normalized.reshape(num_patches, height, width, components)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


# Saves the transformer object
def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Args:
        file_path (str): Path where the object will be saved.
        obj: Object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

# Splits the data into training and testing sets
def preprocess_and_split(patches, labels, test_size=0.2, batch_size=32,ae=False):
    """
    Preprocess hyperspectral data and split into train/test sets.

    Args:
        patches (ndarray): Extracted patches of shape (num_patches, patch_size, patch_size, C).
        labels (ndarray, optional): Ground-truth labels (for supervised learning).
        test_size (float): Fraction of data to use for testing.
        batch_size (int): Batch size for TensorFlow datasets.

    Returns:
        train_dataset, test_dataset: TensorFlow datasets for training and testing.
    """
    if patches.shape[0] != labels.shape[0]:
        raise ValueError("Mismatch between patches and labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        patches, labels, test_size=test_size, stratify=labels, random_state=42
    )

    if ae:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (X_train, y_train)))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, (X_test, y_test)))
        project_logger.info("AutoEncoder Workflow TF datasets successfully created.")
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        project_logger.info("CNN Workflow TF datasets successfully created.")


    train_dataset = train_dataset.shuffle(len(y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset, X_train, X_test, y_train, y_test

# Loads the transformer object
def load_transformer(transformer_path):
    """Loads a transformer object (PCA, Scaler) from a pickle file."""
    try:
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"Transformer file not found at path: {transformer_path}")
        
        with open(transformer_path, 'rb') as file:
            transformer = pickle.load(file)
        
        project_logger.info(f"Transformer object loaded from {transformer_path}.")
        return transformer
    except Exception as e:
        project_logger.error(f"Error occurred while loading transformer object: {e}")
        raise CustomException(e, sys)


################################ USED IN DATA TRANSFORMATION ########################################

################################ USED IN MODEL TRAINING #############################################
def train_model(model, train_dataset, test_dataset, save_model_path, epochs, monitor_metric='val_loss', callbacks=None):
    """
    Train a TensorFlow model.
    
    Args:
        model: The TensorFlow Keras model to train.
        train_dataset: Training dataset.
        test_dataset: Validation/testing dataset.
        save_model_path (str): Path to save the trained model.
        epochs (int): Number of epochs to train for.
        callbacks (list, optional): List of Keras callbacks to use during training.
        
    Returns:
        history: The training history.
    """
    
    # Create default callbacks if none provided
    if callbacks is None:
        callbacks = [
            # Early stopping to prevent overfitting
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                mode='max',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Model checkpoint to save the best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_model_path,
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
        ]
    
    # Train the model
    project_logger.info(f"Starting model training for {epochs} epochs")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model if it wasn't saved by callbacks
    if not any(isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks):
        model.save(save_model_path)
        project_logger.info(f"Model saved to {save_model_path}")
    
    return history

################################ USED IN MODEL TRAINING #############################################

################################ USED IN MODEL PREDICTION ##########################################

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def plot_label_comparison_spatial(y_true, y_pred, label_img_shape, save_path, num_classes, labels_flat=None):
    """
    Plot side-by-side spatial maps of ground truth and predicted labels.
    y_true, y_pred: arrays of valid pixels (flattened)
    label_img_shape: shape of the original label image (H, W)
    labels_flat: flattened original label image (optional, for valid pixel mapping)
    """
    import logging
    mask_flat = label_img_shape[0] * label_img_shape[1]
    if labels_flat is None:
        project_logger.error("labels_flat (flattened original label image) must be provided for correct mapping.")
        raise ValueError("labels_flat (flattened original label image) must be provided for correct mapping.")
    valid_pixel_indices = np.where(labels_flat > 0)[0]
    project_logger.info(f"[Plot] Number of valid pixels: {len(valid_pixel_indices)}")
    project_logger.info(f"[Plot] y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    if len(y_true) != len(valid_pixel_indices) or len(y_pred) != len(valid_pixel_indices):
        project_logger.error(f"[Plot] Shape mismatch: y_true/y_pred ({len(y_true)}) vs valid pixels ({len(valid_pixel_indices)})")
        raise ValueError("Shape mismatch between valid pixels and y_true/y_pred!")
    # Create flat images
    y_true_img_flat = np.zeros(mask_flat, dtype=int)
    y_pred_img_flat = np.zeros(mask_flat, dtype=int)
    y_true_img_flat[valid_pixel_indices] = y_true
    y_pred_img_flat[valid_pixel_indices] = y_pred
    y_true_img = y_true_img_flat.reshape(label_img_shape)
    y_pred_img = y_pred_img_flat.reshape(label_img_shape)
    project_logger.info(f"[Plot] y_true_img shape: {y_true_img.shape}, y_pred_img shape: {y_pred_img.shape}")
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.05})
    cmap = plt.get_cmap('nipy_spectral', num_classes)
    im0 = axes[0].imshow(y_true_img, cmap=cmap, vmin=0, vmax=num_classes-1)
    axes[0].set_title('Ground Truth', fontsize=16)
    axes[0].axis('off')
    im1 = axes[1].imshow(y_pred_img, cmap=cmap, vmin=0, vmax=num_classes-1)
    axes[1].set_title('Predicted Labels', fontsize=16)
    axes[1].axis('off')
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.set_label('Classes', fontsize=14)
    cbar.set_ticks(np.arange(num_classes))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    project_logger.info(f"[Plot] Side-by-side label comparison plot saved to: {save_path}")

################################# USED IN MODEL PREDICTION ##########################################
    





