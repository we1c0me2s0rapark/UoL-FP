import tensorflow as tf

class Losses:
    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2. * intersection + smooth) / (union + smooth)
        return tf.reduce_mean(1 - dice)

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return 0.5 * bce + 0.5 * Losses.dice_loss(y_true, y_pred)

    @staticmethod
    def combined_loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        d_loss = Losses.dice_loss(y_true, y_pred)
        return 0.5 * bce + 0.5 * d_loss

    @staticmethod
    def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

    @staticmethod
    def focal_tversky_loss(y_true, y_pred, gamma=0.75):
        tversky = Losses.tversky_loss(y_true, y_pred)
        return tf.pow(tversky, gamma)

class Metrics:
    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return (2. * intersection + smooth) / (union + smooth)

    @staticmethod
    def jaccard_index(y_true, y_pred, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return (intersection + smooth) / (union + smooth)