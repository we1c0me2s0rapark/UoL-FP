import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
class HnAdam(Adam):
    """
    HnAdam: A customised variant of the Adam optimiser with dynamic normalisation 
    of the update step based on the L2 norm of the scaled gradient estimate.

    Unlike the standard Adam optimiser, this implementation introduces three 
    additional mechanisms:

    1. **L2 norm clipping**: Prevents exploding gradients by constraining the 
       norm of the update vector to lie within a specified range.
    2. **Weight update capping**: Restricts the maximum per-variable update size 
       to avoid sudden, excessively large parameter jumps.
    3. **Dynamic scaling factor**: Adjusts the update step size smoothly by 
       scaling it according to the ratio between the clipped and unclipped norms. 

    These enhancements improve training stability and encourage more reliable 
    convergence compared with the built-in Adam optimiser (with AMSGrad support).

    Args:
        learning_rate (float): Initial learning rate.
        beta_1 (float): Exponential decay rate for the first moment estimates.
        beta_2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant to avoid division by zero.
        norm_min (float): Minimum allowed L2 norm (lower clip).
        norm_max (float): Maximum allowed L2 norm (upper clip).
        amsgrad (bool): Whether to apply the AMSGrad variant of Adam, which 
            maintains the maximum of past squared gradients to improve 
            convergence guarantees.
        max_update_step (float): Optional cap on the maximum update applied to 
            each parameter element.
        name (str): Optional name for the optimiser.
    """
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 norm_min=1e-7,
                 norm_max=10.0,
                 max_update_step=None,
                 amsgrad=True,
                 name="HnAdam",
                 **kwargs):
        # Initialise Adam with AMSGrad (improves theoretical convergence guarantees)
        super().__init__(learning_rate=learning_rate,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         epsilon=epsilon,
                         amsgrad=amsgrad,
                         name=name,
                         **kwargs)
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.max_update_step = max_update_step

    def _resource_apply_dense(self, grad, var):
        # Retrieve the first (m) and second (v) moment slots for this variable
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Prepare constants and effective learning rate
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        t = tf.cast(self.iterations + 1, var_dtype)

        # Update biased first and second moment estimates
        m_t = self.beta_1 * m + (1 - self.beta_1) * grad
        v_t = self.beta_2 * v + (1 - self.beta_2) * tf.square(grad)

        # Write updated values back to slots
        m.assign(m_t)
        v.assign(v_t)

        # Compute bias-corrected estimates for stability
        m_hat = m_t / (1 - tf.pow(self.beta_1, t))
        
        # --- AMSGrad support ---
        # If enabled, track the maximum of past v_t values to improve convergence guarantees
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.maximum(vhat, v_t)
            vhat.assign(vhat_t)
            v_hat = vhat_t / (1 - tf.pow(self.beta_2, t))
        else:
            v_hat = v_t / (1 - tf.pow(self.beta_2, t))

        # Standard Adam update direction (before applying enhancements)
        update = m_hat / (tf.sqrt(v_hat) + epsilon_t)

        # --- Enhancement 1: L2 norm clipping ---
        # Constrain the update vector's norm to the safe range [norm_min, norm_max]
        norm = tf.norm(update)
        clipped_norm = tf.clip_by_value(norm, self.norm_min, self.norm_max)

        # --- Enhancement 2: Dynamic scaling factor ---
        # Smoothly rescale the update based on the ratio of clipped to original norm
        lam = clipped_norm / (norm + 1e-7)
        update = update * lam

        # --- Enhancement 3: Weight update capping ---
        # Prevent excessively large parameter jumps by bounding the overall step size
        if self.max_update_step is not None:
            update_norm = tf.norm(update)
            update = update * tf.minimum(1.0, self.max_update_step / (update_norm + 1e-7))

        # Apply the final, scaled and capped update to the variable
        var.assign_sub(lr_t * update)

    def get_config(self):
        # Ensure custom attributes are serialised with the optimiser
        config = super().get_config()
        config.update({
            "norm_min": self.norm_min,
            "norm_max": self.norm_max,
            "max_update_step": self.max_update_step,
        })
        return config
