import os
import glob
import numpy as np
import cv2
import tensorflow as tf
import pydicom as dicom

class ImageProcessor:
    """
    A class to preprocess DICOM images and their corresponding masks
    for medical image segmentation tasks.
    """

    def __init__(self, image_size):
        """
        Initialise the ImageProcessor.

        Args:
            image_size (tuple): Target image size (height, width).
        """
        self.image_size = image_size

    def load_image(self, image_path):
        """
        Load and preprocess a DICOM image.

        Steps:
            - Read and normalize DICOM pixel data.
            - Convert grayscale to RGB.
            - Resize to the target image size.
            - Normalize to [0, 1].
            - Optionally apply Laplacian enhancement.

        Args:
            image_path (str): File path to a DICOM image.

        Returns:
            tf.Tensor: A float32 RGB image tensor of shape (H, W, 3).
        """
        try:
            ds = dicom.dcmread(image_path)
            image = ds.pixel_array.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load DICOM file: {image_path}") from e

        image = (np.maximum(image, 0) / (image.max() + 1e-7)) * 255.0
        image = np.uint8(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_resized = tf.image.resize(
            image_rgb,
            self.image_size,
            method=tf.image.ResizeMethod.BILINEAR
        )

        image_normalized = tf.cast(image_resized, tf.float32) / 255.0
        return self._apply_laplacian_filter(image_normalized)

    def load_mask(self, image_path):
        """
        Load and preprocess a DICOM mask image.

        Steps:
            - Read pixel data.
            - Binarise the mask.
            - Resize using nearest neighbor.
            - Add channel dimension.

        Args:
            image_path (str): File path to a DICOM mask.

        Returns:
            tf.Tensor: A float32 mask tensor of shape (H, W, 1).
        """
        try:
            ds = dicom.dcmread(image_path)
            mask = ds.pixel_array
        except Exception as e:
            raise ValueError(f"Failed to load DICOM mask: {image_path}") from e

        mask = (mask > 0).astype(np.float32)

        if mask.ndim == 2:
            mask = mask[..., np.newaxis]

        mask_resized = tf.image.resize(
            mask,
            self.image_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        return tf.cast(mask_resized, tf.float32)

    def _apply_laplacian_filter(self, img, apply_gaussian=True):
        """
        Apply optional Gaussian blur and Laplacian filtering
        to enhance edges in an image.

        Args:
            img (tf.Tensor): Float image tensor (H, W, C) in [0, 1].
            apply_gaussian (bool): Whether to apply Gaussian blur before Laplacian.

        Returns:
            tf.Tensor: Edge-enhanced float image tensor in [0, 1].
        """
        if not tf.executing_eagerly():
            raise RuntimeError("apply_laplacian requires eager execution for NumPy operations.")

        laplacian_factor = -0.3
        gaussian_factor = -0.15

        img_np = img.numpy()

        if apply_gaussian:
            gaussian_kernel = np.ones((5, 5), np.float32) / 25.0
            blurred = np.stack([
                cv2.filter2D(img_np[..., c], -1, gaussian_kernel)
                for c in range(img_np.shape[-1])
            ], axis=-1)
            img_np += gaussian_factor * blurred

        laplacian_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)

        laplacian = np.stack([
            cv2.filter2D(img_np[..., c], -1, laplacian_kernel)
            for c in range(img_np.shape[-1])
        ], axis=-1)

        sharpened = img_np + laplacian_factor * laplacian
        sharpened = np.clip(sharpened, 0, 1).astype(np.float32)

        return tf.convert_to_tensor(sharpened, dtype=tf.float32)

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        
    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels