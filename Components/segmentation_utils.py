#
# @brief Import all necessary libraries.
#
# TensorFlow is used for tensor operations, model building, and training.
# The Keras serialisation utility ensures that custom objects (losses/metrics) 
# can be saved and loaded with models.
#
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

class Losses:
    """
    A collection of custom loss functions for image segmentation.
    These are particularly useful for addressing class imbalance in medical datasets.
    """

    @staticmethod
    @register_keras_serializable()
    def dice_loss(y_true, y_pred, smooth=1e-6):
        """
        Dice loss, derived from the Dice coefficient, measures the overlap between
        predicted and ground truth segmentation masks.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.
            smooth (float): Smoothing factor to avoid division by zero.

        Returns:
            tf.Tensor: Dice loss value (lower is better).
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        dice = (2. * intersection + smooth) / (union + smooth)
        return tf.reduce_mean(1 - dice)

    @staticmethod
    @register_keras_serializable()
    def bce_dice_loss(y_true, y_pred):
        """
        Hybrid loss combining Binary Cross-Entropy (BCE) and Dice loss.
        Balances pixel-wise classification accuracy (BCE) with region overlap (Dice).

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            tf.Tensor: Combined BCE–Dice loss.
        """
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return 0.5 * bce + 0.5 * Losses.dice_loss(y_true, y_pred)

    @staticmethod
    @register_keras_serializable()
    def combined_loss(y_true, y_pred):
        """
        Alternative implementation of BCE–Dice loss.
        Often improves performance on imbalanced segmentation tasks.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            tf.Tensor: Combined loss value.
        """
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        d_loss = Losses.dice_loss(y_true, y_pred)
        return 0.5 * bce + 0.5 * d_loss

    @staticmethod
    @register_keras_serializable()
    def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        Tversky loss: a generalisation of Dice/Jaccard losses.
        Allows differential weighting of false positives and false negatives.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.
            alpha (float): Weight for false positives.
            beta (float): Weight for false negatives.
            smooth (float): Smoothing factor.

        Returns:
            tf.Tensor: Tversky loss value.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    @staticmethod
    @register_keras_serializable()
    def focal_tversky_loss(y_true, y_pred, gamma=0.75):
        """
        Focal Tversky loss: emphasises harder-to-classify pixels by raising 
        the Tversky loss to a power (gamma). Useful for highly imbalanced data.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.
            gamma (float): Focusing parameter (higher values increase focus on hard cases).

        Returns:
            tf.Tensor: Focal Tversky loss value.
        """
        tversky = Losses.tversky_loss(y_true, y_pred)
        return tf.pow(tversky, gamma)

class Metrics:
    """
    Custom evaluation metrics for segmentation.
    Provide interpretable measures of model performance.
    """

    @staticmethod
    @register_keras_serializable()
    def dice_coef(y_true, y_pred, smooth=1e-6):
        """
        Dice coefficient: measures similarity between prediction and ground truth.

        Args:
            y_true (tf.Tensor): Ground truth mask.
            y_pred (tf.Tensor): Predicted mask (thresholded at 0.5).
            smooth (float): Smoothing factor.

        Returns:
            tf.Tensor: Dice coefficient (higher is better).
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        return (2. * intersection + smooth) / (union + smooth)

    @staticmethod
    @register_keras_serializable()
    def jaccard_index(y_true, y_pred, smooth=1e-6):
        """
        Jaccard index (Intersection-over-Union, IoU): 
        measures overlap between prediction and ground truth.

        Args:
            y_true (tf.Tensor): Ground truth mask.
            y_pred (tf.Tensor): Predicted mask (thresholded at 0.5).
            smooth (float): Smoothing factor.

        Returns:
            tf.Tensor: Jaccard index (higher is better).
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

        return (intersection + smooth) / (union + smooth)

class Similarities:
    """
    Additional similarity and accuracy measures 
    based on pixel-wise and histogram-based comparisons.
    """

    @staticmethod
    @register_keras_serializable()
    def get_similarity_score(ground_truth, prediction):
        """
        Histogram-based similarity score using OpenCV correlation.
        Compares pixel intensity distributions between ground truth and prediction.

        Args:
            ground_truth (tf.Tensor): Ground truth mask.
            prediction (tf.Tensor): Predicted mask.

        Returns:
            float: Similarity score (%) where 100 indicates perfect match.
        """
        gt = ground_truth.numpy().astype('uint8')
        pred = prediction.numpy().astype('uint8')

        if gt.ndim == 3 and gt.shape[-1] == 1: gt = gt.squeeze(-1)
        if pred.ndim == 3 and pred.shape[-1] == 1: pred = pred.squeeze(-1)

        if np.sum(pred) == 0 and np.sum(gt) != 0: 
            return 0.0

        hist_gt = cv2.calcHist([gt], [0], None, [256], [0, 256])
        cv2.normalize(hist_gt, hist_gt, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_pred = cv2.calcHist([pred], [0], None, [256], [0, 256])
        cv2.normalize(hist_pred, hist_pred, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        metric_val = cv2.compareHist(hist_gt, hist_pred, cv2.HISTCMP_CORREL)
        return round(metric_val * 100, 2)

    @staticmethod
    @register_keras_serializable()
    def get_pixel_accuracy(ground_truth, prediction):
        """
        Pixel-wise accuracy: proportion of pixels classified correctly.

        Args:
            ground_truth (tf.Tensor): Ground truth mask.
            prediction (tf.Tensor): Predicted mask.

        Returns:
            float: Pixel-wise accuracy (%) where 100 indicates perfect prediction.
        """
        gt = ground_truth.numpy().astype(bool)
        pred = prediction.numpy().astype(bool)

        gt_flat = gt.flatten()
        pred_flat = pred.flatten()

        correct = np.sum(gt_flat == pred_flat)
        total = gt_flat.size

        accuracy = correct / total
        return round(accuracy * 100, 2)
