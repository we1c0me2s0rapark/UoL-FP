#
# @brief Import all necessary libraries.
#
# This includes TensorFlow for tensor operations and a utility for Keras serialisation.
#
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

class Losses:
    """
    A collection of custom loss functions for image segmentation.
    These are particularly useful for handling class imbalance in medical imaging datasets.
    """
    @staticmethod
    @register_keras_serializable()
    def dice_loss(y_true, y_pred, smooth=1e-6):
        """
        Calculates the Dice loss, a metric for measuring the similarity between
        the predicted and ground truth segmentation masks.

        Args:
            y_true (tf.Tensor): The true segmentation mask.
            y_pred (tf.Tensor): The predicted segmentation mask.
            smooth (float): A small smoothing factor to prevent division by zero.

        Returns:
            tf.Tensor: The Dice loss value.
        """
        # Cast tensors to float32 for consistent calculations.
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate the intersection and union of the two masks.
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        # Calculate the Dice coefficient and return its complement (1 - dice) as the loss.
        dice = (2. * intersection + smooth) / (union + smooth)
        return tf.reduce_mean(1 - dice)

    @staticmethod
    @register_keras_serializable()
    def bce_dice_loss(y_true, y_pred):
        """
        Combines Binary Cross-Entropy (BCE) and Dice loss.
        This provides a balanced loss function that works well for segmentation tasks.

        Args:
            y_true (tf.Tensor): The true segmentation mask.
            y_pred (tf.Tensor): The predicted segmentation mask.

        Returns:
            tf.Tensor: The combined loss value.
        """
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return 0.5 * bce + 0.5 * Losses.dice_loss(y_true, y_pred)

    @staticmethod
    @register_keras_serializable()
    def combined_loss(y_true, y_pred):
        """
        Another implementation of the combined BCE and Dice loss.
        This is a common practice for improving model performance on imbalanced data.

        Args:
            y_true (tf.Tensor): The true segmentation mask.
            y_pred (tf.Tensor): The predicted segmentation mask.

        Returns:
            tf.Tensor: The combined loss value.
        """
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        d_loss = Losses.dice_loss(y_true, y_pred)
        return 0.5 * bce + 0.5 * d_loss

    @staticmethod
    @register_keras_serializable()
    def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        A generalised version of the Dice and Jaccard losses that allows
        for weighting false positives and false negatives differently.

        Args:
            y_true (tf.Tensor): The true segmentation mask.
            y_pred (tf.Tensor): The predicted segmentation mask.
            alpha (float): Weight for false positives.
            beta (float): Weight for false negatives.
            smooth (float): A small smoothing factor.

        Returns:
            tf.Tensor: The Tversky loss value.
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
        A combination of Tversky loss and Focal loss. This helps the model to
        focus on hard-to-classify examples by down-weighting well-classified ones.

        Args:
            y_true (tf.Tensor): The true segmentation mask.
            y_pred (tf.Tensor): The predicted segmentation mask.
            gamma (float): The focusing parameter.

        Returns:
            tf.Tensor: The Focal Tversky loss value.
        """
        tversky = Losses.tversky_loss(y_true, y_pred)
        return tf.pow(tversky, gamma)

class Metrics:
    """
    A collection of custom evaluation metrics for image segmentation.
    These are used to evaluate model performance after training.
    """
    @staticmethod
    @register_keras_serializable()
    def dice_coef(y_true, y_pred, smooth=1e-6):
        """
        Calculates the Dice coefficient, which measures the similarity between
        the predicted and true masks.

        Args:
            y_true (tf.Tensor): The true segmentation mask.
            y_pred (tf.Tensor): The predicted segmentation mask.
            smooth (float): A small smoothing factor.

        Returns:
            tf.Tensor: The Dice coefficient.
        """
        # Cast to float32 and apply a threshold to the predicted mask.
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return (2. * intersection + smooth) / (union + smooth)

    @staticmethod
    @register_keras_serializable()
    def jaccard_index(y_true, y_pred, smooth=1e-6):
        """
        Calculates the Jaccard index (also known as Intersection over Union),
        a metric for evaluating the overlap of segmentation masks.

        Args:
            y_true (tf.Tensor): The true segmentation mask.
            y_pred (tf.Tensor): The predicted segmentation mask.
            smooth (float): A small smoothing factor.

        Returns:
            tf.Tensor: The Jaccard index.
        """
        # Cast to float32 and apply a threshold to the predicted mask.
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)