#
# @brief Import all required libraries.
#
# TensorFlow is used for tensor operations, model construction, and training.
# The Keras serialisation utility ensures that custom objects (losses/metrics) 
# can be saved together with models and correctly reloaded later.
#
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

class LossFunctions:
    """
    A collection of custom loss functions for medical image segmentation.
    These are particularly useful for handling class imbalance, 
    which is common in medical datasets.
    """

    @staticmethod
    @register_keras_serializable()
    def dice_loss(y_true, y_pred, smooth=1e-6):
        """
        Dice loss, derived from the Dice coefficient, quantifies the overlap 
        between predicted and ground truth segmentation masks.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.
            smooth (float): Smoothing factor to avoid division by zero.

        Returns:
            tf.Tensor: Dice loss value (lower values indicate better performance).
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
        Hybrid loss that combines Binary Cross-Entropy (BCE) with Dice loss.
        This balances pixel-wise classification accuracy (BCE) with region-level 
        overlap quality (Dice).

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            tf.Tensor: Combined BCE–Dice loss.
        """
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return 0.5 * bce + 0.5 * LossFunctions.dice_loss(y_true, y_pred)

    @staticmethod
    @register_keras_serializable()
    def combined_loss(y_true, y_pred):
        """
        Alternative implementation of BCE–Dice hybrid loss.
        This formulation is often more effective on imbalanced segmentation tasks.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.

        Returns:
            tf.Tensor: Combined loss value.
        """
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        d_loss = LossFunctions.dice_loss(y_true, y_pred)
        return 0.5 * bce + 0.5 * d_loss

    @staticmethod
    @register_keras_serializable()
    def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        Tversky loss: a generalisation of Dice/Jaccard losses.
        Provides adjustable weighting for false positives and false negatives.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.
            alpha (float): Weight applied to false positives.
            beta (float): Weight applied to false negatives.
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
        Focal Tversky loss: places greater emphasis on harder-to-classify pixels 
        by raising the Tversky loss to the power of gamma. Particularly suitable 
        for highly imbalanced datasets.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask.
            y_pred (tf.Tensor): Predicted segmentation mask.
            gamma (float): Focusing parameter (higher values place more emphasis on difficult cases).

        Returns:
            tf.Tensor: Focal Tversky loss value.
        """
        tversky = LossFunctions.tversky_loss(y_true, y_pred)
        return tf.pow(tversky, gamma)

class SegmentationMetrics:
    """
    Custom evaluation metrics for segmentation tasks.
    Provide interpretable and clinically meaningful measures of performance.
    """

    @staticmethod
    @register_keras_serializable()
    def dice_coef(y_true, y_pred, smooth=1e-6):
        """
        Dice coefficient: quantifies similarity between predicted and ground truth masks.

        Args:
            y_true (tf.Tensor): Ground truth mask.
            y_pred (tf.Tensor): Predicted mask (thresholded at 0.5).
            smooth (float): Smoothing factor.

        Returns:
            tf.Tensor: Dice coefficient (higher values indicate better segmentation quality).
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
        measures the degree of overlap between predicted and ground truth masks.

        Args:
            y_true (tf.Tensor): Ground truth mask.
            y_pred (tf.Tensor): Predicted mask (thresholded at 0.5).
            smooth (float): Smoothing factor.

        Returns:
            tf.Tensor: Jaccard index (higher values indicate better segmentation quality).
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

        return (intersection + smooth) / (union + smooth)

class SimilarityMetrics:
    """
    Global similarity indicators based on both pixel-wise and histogram-based comparisons.
    """

    @staticmethod
    @register_keras_serializable()
    def get_histogram_similarity(ground_truth, prediction):
        """
        Histogram-based similarity score using OpenCV correlation.
        Compares the intensity distributions of ground truth and predicted masks.

        Args:
            ground_truth (tf.Tensor): Ground truth mask.
            prediction (tf.Tensor): Predicted mask.

        Returns:
            float: Histogram similarity score (percentage). 
                   100 indicates perfect similarity.
        """
        gt = ground_truth.numpy().astype('uint8')
        pred = prediction.numpy().astype('uint8')

        if gt.ndim == 3 and gt.shape[-1] == 1: 
            gt = gt.squeeze(-1)
        if pred.ndim == 3 and pred.shape[-1] == 1: 
            pred = pred.squeeze(-1)

        # If prediction is empty but ground truth is not, similarity is zero
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
        Pixel-wise accuracy: proportion of pixels classified correctly 
        compared with the ground truth.

        Args:
            ground_truth (tf.Tensor): Ground truth mask.
            prediction (tf.Tensor): Predicted mask.

        Returns:
            float: Pixel-wise accuracy (percentage). 
                   100 indicates a perfect prediction.
        """
        gt = ground_truth.numpy().astype(bool)
        pred = prediction.numpy().astype(bool)

        gt_flat = gt.flatten()
        pred_flat = pred.flatten()

        correct = np.sum(gt_flat == pred_flat)
        total = gt_flat.size

        accuracy = correct / total
        return round(accuracy * 100, 2)
