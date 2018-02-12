import tensorflow as tf
from utils.ops import *


def generator(x, c, reuse=False):
    with tf.variable_scope("generator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        c = tf.expand_dims(tf.expand_dims(c, 1), 1)
        c = tf.tile(c, [1, x.get_shape()[1].value, x.get_shape()[2].value, 1])
        x = tf.concat([x, c], axis=3)

        x = relu(instance_norm(conv2d(x, 64, kernel_size=7, strides=[1, 1, 1, 1], name="gen_ds_conv1"), name="in1_1"))
        x = relu(instance_norm(conv2d(x, 128, kernel_size=4, strides=[1, 2, 2, 1], name="gen_ds_conv2"), name="in1_2"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=4, strides=[1, 2, 2, 1], name="gen_ds_conv3"), name="in1_3"))

        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv1"), name="in2_1"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv2"), name="in2_2"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv3"), name="in2_3"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv4"), name="in2_4"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv5"), name="in2_5"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv6"), name="in2_6"))

        x = relu(
            instance_norm(deconv_2d(x, [32, 14, 14, 128], kernel_size=4, strides=[1, 2, 2, 1], name="gen_us_conv1"),
                          name="in3_1"))
        x = relu(
            instance_norm(deconv_2d(x, [32, 28, 28, 64], kernel_size=4, strides=[1, 2, 2, 1], name="gen_us_conv2"),
                          name="in3_2"))
        x = tanh(conv2d(x, 3, kernel_size=7, strides=[1, 1, 1, 1], name="gen_us_conv3"))
        return x





