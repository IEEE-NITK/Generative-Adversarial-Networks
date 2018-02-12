import tensorflow as tf
from utils.ops import *


def discriminator(x, nd, reuse=False):
    with tf.variable_scope("generator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        size = tf.shape(x)[1]

        x = lrelu(conv2d(x, 64, kernel_size=4, strides=[1, 2, 2, 1], name="dis_conv1"))

        x = lrelu(conv2d(x, 128, kernel_size=4, strides=[1, 2, 2, 1], name="dis_conv2"))
        x = lrelu(conv2d(x, 256, kernel_size=4, strides=[1, 2, 2, 1], name="dis_conv3"))
        x = lrelu(conv2d(x, 512, kernel_size=4, strides=[1, 2, 2, 1], name="dis_conv4"))
        x = lrelu(conv2d(x, 1024, kernel_size=4, strides=[1, 2, 2, 1], name="dis_conv5"))
        x = lrelu(conv2d(x, 2048, kernel_size=4, strides=[1, 2, 2, 1], name="dis_conv6"))

        d_src = conv2d(x, 1, kernel_size=3, strides=[1, 1, 1, 1], name="d_src")
        d_cls = conv2d(x, nd, kernel_size=size/64, strides=[1, 1, 1, 1], name="d_cls")

        return d_src, d_cls