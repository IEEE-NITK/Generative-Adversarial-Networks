"""
Some methods are from https://github.com/carpedm/DCGAN-tensorflow
"""
import numpy as np
import scipy.misc
import tensorflow as tf
import math


def conv_layer(x, output, name="conv"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [5, 5, x.get_shape()[-1], output], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding="SAME") + b


def conv_transpose_layer(x, output_shape, name="conv_transpose"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [5, 5, output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, 2, 2, 1]) + b


def leaky_relu(x, leak=0.2, name="leaky_relu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def sigmoid_cross_entropy(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)
    

def dense(x, output, scope=None):
    with tf.variable_scope(scope or "linear"):
        w = tf.get_variable('w', [x.get_shape()[-1], output], tf.float32, tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b


def conv_cond_concat(x, y):
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge_images(images, size):
    return inverse_transform(images)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def inverse_transform(images):
    return (images+1) / 2.


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w
