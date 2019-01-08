import tensorflow as tf
from utils.ops import *


def generator2(x, c, reuse=False):
    with tf.variable_scope("generator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        c = tf.expand_dims(tf.expand_dims(c, 1), 1)
        c = tf.tile(c, [1, x.get_shape()[1].value, x.get_shape()[2].value, 1])
        x = tf.concat([x, c], axis=3)
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='CONSTANT')
        x = relu(instance_norm(conv2d(x, 64, kernel_size=7, strides=[1, 1, 1, 1], name="gen_ds_conv1"), name="gen_in1_1"))
        x = relu(instance_norm(conv2d(x, 128, kernel_size=4, strides=[1, 2, 2, 1], name="gen_ds_conv2"), name="gen_in1_2"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=4, strides=[1, 2, 2, 1], name="gen_ds_conv3"), name="gen_in1_3"))

        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv1"), name="gen_in2_1"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv2"), name="gen_in2_2"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv3"), name="gen_in2_3"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv4"), name="gen_in2_4"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv5"), name="gen_in2_5"))
        x = relu(instance_norm(conv2d(x, 256, kernel_size=3, strides=[1, 1, 1, 1], name="gen_bn_conv6"), name="gen_in2_6"))

        x = relu(
            instance_norm(deconv_2d(x, [16, 64, 64, 128], kernel_size=4, strides=[1, 2, 2, 1], name="gen_us_conv1"),
                          name="gen_in3_1"))
        x = relu(
            instance_norm(deconv_2d(x, [16, 128, 128, 64], kernel_size=4, strides=[1, 2, 2, 1], name="gen_us_conv2"),
                          name="gen_in3_2"))
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="CONSTANT")
        x = tanh(conv2d(x, 3, kernel_size=7, strides=[1, 1, 1, 1], name="gen_us_conv3", padding='VALID'))

        return x


def generator(image, labels, reuse=False, output_channels=3, repeat_num=6):
    with tf.variable_scope("generator"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        layer_specs_encoder = [
            (64, 7, 1, 3),
            (128, 4, 2, 1),
            (256, 4, 2, 1)
        ]
        layer_specs_decoder = [
            (128, 4, 2, 1),
            (64, 4, 2, 1)
        ]

        layers = []

        c = tf.expand_dims(tf.expand_dims(labels, 1), 2)
        c = tf.tile(c, [1, tf.shape(image)[1], tf.shape(image)[2], 1])
        inputs = tf.concat([image, c], axis=3)
        # TODO shape inference
        inputs.set_shape([16, 128, 128, 8])
        print(c)
        print(image)
        print(inputs)
        layers.append(inputs)
        # Encoder
        for (filters, kernel_size, stride, pad) in layer_specs_encoder:
            with tf.variable_scope("generator_encoder_%d" % (len(layers))):
                padded_input = tf.pad(layers[-1], [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="CONSTANT")
                convolved = tf.layers.conv2d(padded_input, filters, kernel_size, stride, padding="VALID",
                                             use_bias=False)
                convolved = tf.nn.relu(convolved)
                #             output = instance_norm(convolved)
                layers.append(convolved)

        # Residual Block
        for i in range(repeat_num):
            with tf.variable_scope("generator_res_%d" % (i + 1)):
                padded_input = tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
                c1 = tf.layers.conv2d(padded_input, 256, 3, 1, padding="VALID", use_bias=False)
                c1 = tf.nn.relu(c1)
                #             in1 = instance_norm(c1)
                p1 = tf.pad(c1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
                c2 = tf.layers.conv2d(p1, 256, 3, 1, padding="VALID", activation=None, use_bias=False)
                #             in2 = instance_norm(c2)
                res = layers[-1] + c2
                layers.append(res)

        # Decoder
        for (filters, kernel_size, stride, pad) in layer_specs_decoder:
            with tf.variable_scope("generator_decoder_%d" % (len(layers))):
                convolved = deconv(layers[-1], filters)
                #             output = instance_norm(convolved)
                layers.append(convolved)

        with tf.variable_scope("generator_decoder_out"):
            padded_input = tf.pad(layers[-1], [[0, 0], [3, 3], [3, 3], [0, 0]], mode="CONSTANT")
            convolved = tf.layers.conv2d(padded_input, output_channels, 7, 1, padding="VALID", use_bias=False)
            convolved = tf.tanh(convolved)
            layers.append(convolved)

        return convolved
