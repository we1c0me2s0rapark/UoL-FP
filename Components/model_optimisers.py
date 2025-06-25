import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
class HnAdam(Adam):
    """
    HnAdam: A variant of the Adam optimizer with dynamic normalization of the update step
    based on the norm of the scaled gradient estimate.

    Args:
        learning_rate (float): Initial learning rate.
        beta_1 (float): Exponential decay rate for the first moment estimates.
        beta_2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant to avoid division by zero.
        amsgrad (bool): Whether to apply the AMSGrad variant of this algorithm.
        norm_min (float): Minimum allowed norm value (lower clip).
        norm_max (float): Maximum allowed norm value (upper clip).
        name (str): Optional name for the optimizer.
    """
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 norm_min=1e-7,
                 norm_max=10.0,
                 name="HnAdam",
                 **kwargs):
        super().__init__(learning_rate=learning_rate,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         epsilon=epsilon,
                         amsgrad=amsgrad,
                         name=name,
                         **kwargs)
        self.norm_min = norm_min
        self.norm_max = norm_max

    def _resource_apply_dense(self, grad, var):
        # Perform standard Adam update to ensure m, v are updated
        super()._resource_apply_dense(grad, var)

        # Retrieve the m and v slots
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        var_dtype = var.dtype.base_dtype
        t = tf.cast(self.iterations + 1, var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        lr_t = self._decayed_lr(var_dtype)

        # Compute bias-corrected moments
        m_hat = m / (1 - tf.pow(self.beta_1, t))
        v_hat = v / (1 - tf.pow(self.beta_2, t))

        # Compute the norm of the scaled update
        update = m_hat / (tf.sqrt(v_hat) + epsilon_t)
        norm = tf.norm(update)
        clipped_norm = tf.clip_by_value(norm, self.norm_min, self.norm_max)
        lam = clipped_norm / 2.0

        # Apply the scaled update to the variable
        var.assign_sub(lr_t * update * lam)

    def get_config(self):
        config = super().get_config()
        config.update({
            "norm_min": self.norm_min,
            "norm_max": self.norm_max,
        })
        return config
