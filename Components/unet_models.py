import tensorflow as tf
from tensorflow.keras import layers, models, applications
from keras.saving import register_keras_serializable

#
# @brief A collection of utility functions and custom layers for building complex
# deep learning architectures, particularly U-Net and its variants.
#

class ResizeToMatch(layers.Layer):
    """
    A custom Keras layer to resize a source tensor to match the spatial dimensions
    of a target tensor. This is useful for creating skip connections where feature maps
    have different resolutions.
    """
    def call(self, inputs):
        """
        Resizes the source tensor to match the spatial dimensions of the target tensor.

        Args:
            inputs (list): A list containing two tensors:
                - source (tf.Tensor): The tensor to be resized.
                - target (tf.Tensor): The tensor whose shape to match.

        Returns:
            tf.Tensor: The resized source tensor.
        """
        source, target = inputs
        # Get the height and width from the target tensor's shape.
        target_shape = tf.shape(target)
        # Resize the source tensor to the target's height and width using bilinear interpolation.
        return tf.image.resize(source, (target_shape[1], target_shape[2]), method='bilinear')

def double_conv_block(x, filters, padding="same"):
    """
    A standard double 2D convolutional block, commonly used in U-Net architectures.

    This block consists of two convolutional layers with ReLU activation, which
    helps in feature extraction.

    Args:
        x (tf.Tensor): The input tensor.
        filters (int): The number of filters for the convolutional layers.
        padding (str): Padding type, typically "same" to preserve spatial dimensions.

    Returns:
        tf.Tensor: The output of the double convolutional block.
    """
    x = layers.Conv2D(filters, 3, padding=padding, activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.Conv2D(filters, 3, padding=padding, activation='relu',
                     kernel_initializer='he_normal')(x)
    return x

def downsample_block(x, filters, padding="same"):
    """
    A downsampling block for the U-Net encoder path.

    It applies a double convolutional block followed by max-pooling to reduce
    the spatial dimensions and increase the number of feature maps.

    Args:
        x (tf.Tensor): The input tensor.
        filters (int): The number of filters for the convolutional layers.
        padding (str): Padding type, typically "same".

    Returns:
        tuple: A tuple containing two tensors:
            - The feature map before max-pooling (for the skip connection).
            - The downsampled output.
    """
    f = double_conv_block(x, filters, padding)
    p = layers.MaxPool2D(pool_size=(2, 2))(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x, skip, filters, padding="same"):
    """
    An upsampling block for the U-Net decoder path.

    It uses a transposed convolution to increase spatial dimensions, concatenates
    the result with a skip connection from the encoder, and then applies a
    double convolutional block.

    Args:
        x (tf.Tensor): The input tensor from the previous decoder block.
        skip (tf.Tensor): The feature map from the corresponding encoder block
                          for the skip connection.
        filters (int): The number of filters for the convolutional layers.
        padding (str): Padding type, typically "same".

    Returns:
        tf.Tensor: The output of the upsampling block.
    """
    # Use Conv2DTranspose for upsampling
    x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2,
                               padding=padding, activation='relu')(x)
    # Concatenate the upsampled tensor with the skip connection
    x = layers.Concatenate()([x, skip])
    x = layers.Dropout(0.3)(x)
    # Apply a double convolutional block
    x = double_conv_block(x, filters, padding)
    return x

def ASPP(x, filters):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.

    This module captures multi-scale context by applying parallel atrous convolutions
    with different dilation rates, and then concatenating their outputs. This is
    useful for handling objects of various sizes.

    Args:
        x (tf.Tensor): The input tensor, typically from the bottleneck of the encoder.
        filters (int): The number of filters for the convolutional layers in the ASPP module.

    Returns:
        tf.Tensor: The output tensor with enhanced multi-scale context.
    """
    # Atrous convolutions with different dilation rates.
    y1 = layers.Conv2D(filters, 1, padding="same", activation='relu')(x)
    y2 = layers.Conv2D(filters, 3, dilation_rate=6, padding="same", activation='relu')(x)
    y3 = layers.Conv2D(filters, 3, dilation_rate=12, padding="same", activation='relu')(x)
    y4 = layers.Conv2D(filters, 3, dilation_rate=18, padding="same", activation='relu')(x)

    # Image-level features with global average pooling.
    y5 = layers.GlobalAveragePooling2D()(x)
    y5 = layers.Reshape((1, 1, y5.shape[-1]))(y5)
    y5 = layers.Conv2D(filters, 1, padding="same", activation='relu')(y5)
    y5 = ResizeToMatch()([y5, x])

    # Concatenate all features and apply a final convolution.
    y = layers.Concatenate()([y1, y2, y3, y4, y5])
    return layers.Conv2D(filters, 1, padding="same", activation='relu')(y)

def decoder_block(x, skip, filters):
    """
    A decoder block for the Double U-Net, which combines a transposed convolution,
    a resized skip connection, and a double convolutional block.

    Args:
        x (tf.Tensor): The input tensor from the previous decoder block.
        skip (tf.Tensor): The feature map from the corresponding encoder block.
        filters (int): The number of filters for the convolutional layers.

    Returns:
        tf.Tensor: The output of the decoder block.
    """
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    # Resize the skip connection to match the upsampled tensor.
    skip_resized = ResizeToMatch()([skip, x])
    # Concatenate the tensors and apply convolutions.
    x = layers.Concatenate()([x, skip_resized])
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

# ========== U-Net Model Class ==========

@register_keras_serializable(package="Custom")
class ModifiedUNet(tf.keras.Model):
    """
    A standard U-Net model architecture for image segmentation, implemented as a
    Keras Model subclass for flexibility and customisation.

    The U-Net is an encoder-decoder network with skip connections that are crucial
    for preserving spatial information and improving segmentation accuracy.
    """
    def __init__(self, input_shape=(256, 256, 3), name=None, **kwargs):
        """
        Initialises the Modified U-Net model.

        Args:
            input_shape (tuple): The shape of the input images (height, width, channels).
            name (str, optional): The name of the model.
        """
        super(ModifiedUNet, self).__init__(name=name, **kwargs)
        self.input_shape_ = input_shape
        self.model = None

    def build(self, input_shape):
        """
        Builds the underlying Keras model instance. This is called automatically
        by Keras the first time the model is used.
        """
        if self.model is None:
            self.model = self.build_model()
        super().build(input_shape)

    def build_model(self):
        """
        Defines the U-Net architecture.

        Returns:
            tf.keras.Model: The constructed U-Net model.
        """
        inputs = layers.Input(shape=self.input_shape_)
        filters = [64, 128, 256, 512, 1024]

        # Encoder (Contracting Path)
        # Each block downsamples the feature map and stores a skip connection.
        f1, p1 = downsample_block(inputs, filters[0])
        f2, p2 = downsample_block(p1, filters[1])
        f3, p3 = downsample_block(p2, filters[2])
        f4, p4 = downsample_block(p3, filters[3])

        # Bottleneck
        # This is the deepest part of the network, capturing high-level features.
        bottleneck = double_conv_block(p4, filters[4])

        # Decoder (Expansive Path) with skip connections
        # The upsampling blocks increase the spatial resolution while incorporating
        # features from the encoder via skip connections.
        u6 = upsample_block(bottleneck, f4, filters[3])
        u7 = upsample_block(u6, f3, filters[2])
        u8 = upsample_block(u7, f2, filters[1])
        u9 = upsample_block(u8, f1, filters[0])

        # Final output layer
        # A 1x1 convolution with a sigmoid activation produces the final
        # segmentation mask with values between 0 and 1.
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(u9)

        return models.Model(inputs, outputs, name="Modified_U-Net")

    def call(self, inputs):
        """
        Defines the forward pass of the model. It simply calls the underlying
        built model with the given inputs.
        """
        if self.model is None:
            self.build(inputs.shape)
        return self.model(inputs)

    def get_config(self):
        """
        Returns the configuration of the layer for serialisation. This allows
        the model to be saved and loaded correctly.
        """
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Creates a layer instance from its configuration. This is used by Keras
        when loading a model from a saved file.
        """
        print(f"Config passed to from_config: {config}")
        
        # This part is still necessary for custom Model subclasses,
        # even if not using @register_keras_serializable, because
        # Keras needs to know how to rebuild the 'internal_model'
        # which isn't automatically handled by default.
        input_shape = config.pop("input_shape", (128, 128, 3))
        instance = cls(input_shape=input_shape, **config)
        return instance