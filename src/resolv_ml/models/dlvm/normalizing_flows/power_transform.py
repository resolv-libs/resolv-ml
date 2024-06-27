import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import tf_keras


def power_transform_bijector():
    shift_bij = tfp.bijectors.Shift(
        shift=tf.Variable(initial_value=0.0, trainable=True, name='shift'),
        validate_args=False,
        name='shift'
    )
    pt_bij = tfp.bijectors.PowerTransform(
        power=tf.Variable(initial_value=0.0, trainable=True, name='power'),
        validate_args=False,
        parameters=None,
        name='power_transform'
    )
    batch_norm_bij = tfp.bijectors.BatchNormalization(
        batchnorm_layer=tf_keras.layers.BatchNormalization(),
        training=True,
        validate_args=False,
        name='batch_norm'
    )
    return tfp.bijectors.Chain(
        [batch_norm_bij, pt_bij, shift_bij],
        validate_args=False,
        validate_event_size=False,
        parameters=None
    )


if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow_probability import bijectors as tfb
    from tensorflow_probability import distributions as tfd

    # A common choice for a normalizing flow is to use a Gaussian for the base
    # distribution. (However, any continuous distribution would work.) E.g.,
    nvp = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(
            loc=tf.random.normal(shape=(32, 256)),
            scale_diag=tf.random.normal(shape=(32, 256))
        ),
        bijector=tfb.RealNVP(
            num_masked=0,
            shift_and_log_scale_fn=tfb.real_nvp_default_template(
                hidden_layers=[512, 512])
        )
    )

    post_multivariate_normal = tfd.MultivariateNormalDiag(
        loc=tf.random.normal(shape=(32, 256)),
        scale_diag=tf.random.normal(shape=(32, 256))
    )

    joint_post = tfd.Independent([nvp, post_multivariate_normal])

    x = nvp.sample()
    nvp.log_prob(x)
    nvp.log_prob([0.0, 0.0, 0.0])
