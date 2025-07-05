import tensorflow as tf
from tensorflow.keras import layers, models, applications
from keras.saving import register_keras_serializable

# ========= Utility Functions and Layers ==========

class ResizeToMatch(layers.Layer):
    def call(self, inputs):
        source, target = inputs
        target_shape = tf.shape(target)
        return tf.image.resize(source, (target_shape[1], target_shape[2]), method='bilinear')

def double_conv_block(x, filters, padding="same"):
    x = layers.Conv2D(filters, 3, padding=padding, activation='relu',
                      kernel_initializer='he_normal')(x)
    x = layers.Conv2D(filters, 3, padding=padding, activation='relu',
                      kernel_initializer='he_normal')(x)
    return x

def downsample_block(x, filters, padding="same"):
    f = double_conv_block(x, filters, padding)
    p = layers.MaxPool2D(pool_size=(2, 2))(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x, skip, filters, padding="same"):
    x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2,
                               padding=padding, activation='relu')(x)
    x = layers.Concatenate()([x, skip])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, filters, padding)
    return x

def ASPP(x, filters):
    y1 = layers.Conv2D(filters, 1, padding="same", activation='relu')(x)
    y2 = layers.Conv2D(filters, 3, dilation_rate=6, padding="same", activation='relu')(x)
    y3 = layers.Conv2D(filters, 3, dilation_rate=12, padding="same", activation='relu')(x)
    y4 = layers.Conv2D(filters, 3, dilation_rate=18, padding="same", activation='relu')(x)

    y5 = layers.GlobalAveragePooling2D()(x)
    y5 = layers.Reshape((1, 1, y5.shape[-1]))(y5)
    y5 = layers.Conv2D(filters, 1, padding="same", activation='relu')(y5)
    y5 = ResizeToMatch()([y5, x])

    y = layers.Concatenate()([y1, y2, y3, y4, y5])
    return layers.Conv2D(filters, 1, padding="same", activation='relu')(y)

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    skip_resized = ResizeToMatch()([skip, x])
    x = layers.Concatenate()([x, skip_resized])
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

# ========== U-Net Model Class ==========

@register_keras_serializable(package="Custom")
class ModifiedUNet(tf.keras.Model):
    def __init__(self, input_shape=(256, 256, 3), name=None, **kwargs):
        super(ModifiedUNet, self).__init__(name=name, **kwargs)
        self.input_shape_ = input_shape
        self.model = None

    def build(self, input_shape):
        if self.model is None:
            self.model = self.build_model()
        super().build(input_shape)

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape_)
        filters = [64, 128, 256, 512, 1024]

        f1, p1 = downsample_block(inputs, filters[0])
        f2, p2 = downsample_block(p1, filters[1])
        f3, p3 = downsample_block(p2, filters[2])
        f4, p4 = downsample_block(p3, filters[3])

        bottleneck = double_conv_block(p4, filters[4])

        u6 = upsample_block(bottleneck, f4, filters[3])
        u7 = upsample_block(u6, f3, filters[2])
        u8 = upsample_block(u7, f2, filters[1])
        u9 = upsample_block(u8, f1, filters[0])

        outputs = layers.Conv2D(1, 1, activation='sigmoid')(u9)

        return models.Model(inputs, outputs, name="Modified_U-Net")

    def call(self, inputs):
        if self.model is None:
            self.build(inputs.shape)
        return self.model(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        print(f"Config passed to from_config: {config}")
        
        # This part is still necessary for custom Model subclasses,
        # even if not using @register_keras_serializable, because
        # Keras needs to know how to rebuild the 'internal_model'
        # which isn't automatically handled by default.
        input_shape = config.pop("input_shape", (128, 128, 3))
        instance = cls(input_shape=input_shape, **config)
        return instance

# ========== Double U-Net Model Class ==========

@register_keras_serializable(package="Custom")
class DoubleUNet(tf.keras.Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super(DoubleUNet, self).__init__()
        self.input_shape_ = input_shape
        self.model = self.build_model()

    def build_encoder(self, name):
        inputs = layers.Input(shape=self.input_shape_)
        if name == "xception":
            base = applications.Xception(include_top=False, weights="imagenet", input_tensor=inputs)
            skips = [base.get_layer(n).output for n in [
                "block1_conv1_act", "block3_sepconv2_act", "block4_sepconv2_act", "block13_sepconv2_act"
            ]]
            output = base.get_layer("block14_sepconv2_act").output
        elif name == "densenet":
            base = applications.DenseNet121(include_top=False, weights="imagenet", input_tensor=inputs)
            skips = [base.get_layer(n).output for n in [
                "conv1_relu", "pool2_relu", "pool3_relu", "pool4_relu"
            ]]
            output = base.get_layer("relu").output
        elif name == "vgg19":
            base = applications.VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
            skips = [base.get_layer(n).output for n in [
                "block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4"
            ]]
            output = base.get_layer("block5_conv4").output
        else:
            raise ValueError(f"Unsupported encoder: {name}")
        return models.Model(inputs=inputs, outputs=[skips, output], name=f"{name}_encoder")

    def build_decoder(self, x, skips):
        x = ASPP(x, 256)
        x = decoder_block(x, skips[-1], 256)
        x = decoder_block(x, skips[-2], 128)
        x = decoder_block(x, skips[-3], 64)
        x = decoder_block(x, skips[-4], 32)
        return layers.Conv2D(1, 1, activation="sigmoid")(x)

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape_)
        size = (self.input_shape_[0], self.input_shape_[1])

        encoders = {
            'x': self.build_encoder("xception"),
            'd': self.build_encoder("densenet"),
            'v': self.build_encoder("vgg19")
        }

        # === First Pass ===
        skips_x, out_x = encoders['x'](inputs)
        skips_d, out_d = encoders['d'](inputs)
        skips_v, out_v = encoders['v'](inputs)

        out_x_dec = self.build_decoder(out_x, skips_x)
        out_d_dec = self.build_decoder(out_d, skips_d)
        out_v_dec = self.build_decoder(out_v, skips_v)

        out_x_dec = layers.Lambda(lambda t: tf.image.resize(t, size))(out_x_dec)
        out_d_dec = layers.Lambda(lambda t: tf.image.resize(t, size))(out_d_dec)
        out_v_dec = layers.Lambda(lambda t: tf.image.resize(t, size))(out_v_dec)

        out1 = layers.Conv2D(1, 1, activation='sigmoid')(
            layers.Concatenate()([out_x_dec, out_d_dec, out_v_dec])
        )

        # === Second Pass ===
        masked_input = layers.Add()([
            inputs,
            layers.Multiply()([inputs, out1])
        ])

        skips_x2, out_x2 = encoders['x'](masked_input)
        skips_d2, out_d2 = encoders['d'](masked_input)
        skips_v2, out_v2 = encoders['v'](masked_input)

        out_x2_dec = self.build_decoder(out_x2, skips_x2)
        out_d2_dec = self.build_decoder(out_d2, skips_d2)
        out_v2_dec = self.build_decoder(out_v2, skips_v2)

        out_x2_dec = layers.Lambda(lambda t: tf.image.resize(t, size))(out_x2_dec)
        out_d2_dec = layers.Lambda(lambda t: tf.image.resize(t, size))(out_d2_dec)
        out_v2_dec = layers.Lambda(lambda t: tf.image.resize(t, size))(out_v2_dec)

        out2 = layers.Conv2D(1, 1, activation='sigmoid')(
            layers.Concatenate()([out_x2_dec, out_d2_dec, out_v2_dec])
        )

        final = layers.Conv2D(1, 1, activation='sigmoid')(layers.Concatenate()([out1, out2]))

        return models.Model(inputs=inputs, outputs=final, name="Double_U-Net")

    def call(self, inputs):
        return self.model(inputs)
