import tensorflow as tf
from utils.discriminator import discriminator


def wgan_gp_loss(real_img, fake_img, labels, lambda_gp, epsilon):
    hat_img = epsilon * real_img + (1. - epsilon) * fake_img
    gradients = tf.gradients(discriminator(hat_img, labels, reuse=True)[0], xs=[hat_img])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))

    return lambda_gp * gradient_penalty


def gan_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def classification_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def recon_loss(image1, image2):
    return tf.reduce_mean(tf.abs(image1 - image2))