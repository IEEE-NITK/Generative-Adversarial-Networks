import tensorflow as tf


def conv2d(x, output, kernel_size, strides, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [kernel_size, kernel_size, x.get_shape()[-1], output],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b',
                            [output],
                            initializer=tf.constant_initializer(0.0))

        return tf.nn.conv2d(x, w, strides=strides, padding='SAME') + b


def deconv_2d(x, output_shape, kernel_size, strides, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [kernel_size, kernel_size, output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b',
                            [output_shape[-1]],
                            initializer=tf.constant_initializer(0.0))

        return tf.nn.conv2d_transpose(x, w,
                                      output_shape=output_shape,
                                      strides=strides,
                                      padding='SAME') + b


def instance_norm(x, name='instance_norm'):
    with tf.variable_scope(name):
        depth = x.get_shape()[-1]

        scale = tf.get_variable('scale', [depth],
                                initializer=tf.random_normal_initializer(1.0, 0.02,
                                                                         dtype=tf.float32))
        offset = tf.get_variable('offset', [depth],
                                 initializer=tf.constant_initializer(0.0))

        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)

        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv

        return scale * normalized + offset


def relu(x, name='relu'):
    return tf.nn.relu(x)


def lrelu(x, leak=0.02, name='lrelu'):
    return tf.maximum(x, leak * x)


def tanh(x, name='tanh'):
    return tf.nn.tanh(x)