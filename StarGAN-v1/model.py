import tensorflow as tf
from utils.ops import *
from data_loader import DataLoader


class StarGAN:
    def __init__(self, sess, config, data):

        # Model hyper-parameters
        self.sess = sess
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size

        # Hyper-parameters
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = tf.Variable(config.beta1)
        self.beta2 = tf.Variable(config.beta2)
        self.x, self.real_labels = data.batch

        self.dataset = config.dataset
        self.num_iters = config.num_iters
        self.batch_size = config.batch_size
        self.fixed_c_list = self.make_celebA_labels(model.real_labels)
        self.fixed_c_list = tf.unstack(self.fixed_c_list, axis=1)
        self.fixed_c_list = tf.stack(self.fixed_c_list, axis=0)

        self.build_model()

    def generator(self, x, c, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            c = tf.expand_dims(tf.expand_dims(c, 1), 1)
            c = tf.tile(c, [1, x.get_shape()[1].value, x.get_shape()[2].value, 1])
            x = tf.concat([x, c], axis=3)

            x = relu(instance_norm(conv2d(x, 64, kernel_size=7, strides=[1, 1, 1, 1], padding=3, name="gen_ds_conv1"),
                                   name="gen_in1_1"))
            x = relu(instance_norm(conv2d(x, 128, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="gen_ds_conv2"),
                                   name="gen_in1_2"))
            x = relu(instance_norm(conv2d(x, 256, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="gen_ds_conv3"),
                                   name="gen_in1_3"))

            x_ = x
            for i in range(6):
                res1 = relu(conv2d(x_, 256, kernel_size=3, strides=[1, 1, 1, 1], padding=1,
                                   name="gen_res_conv{}_0".format(i)))
                res2 = conv2d(res1, 256, kernel_size=3, strides=[1, 1, 1, 1], padding=1,
                              name="gen_res_conv{}_0".format(i))
                x = x_ + res2
                x_ = x

            x = relu(instance_norm(deconv2d(x, [16, 64, 64, 128], kernel_size=4, strides=[1, 2, 2, 1], padding=1,
                                            name="gen_us_deconv1"), name="gen_in3_1"))
            x = relu(instance_norm(deconv2d(x, [16, 128, 128, 64], kernel_size=4, strides=[1, 2, 2, 1], padding=1,
                                            name="gen_us_deconv2"), name="gen_in3_2"))

            x = tanh(conv2d(x, 3, kernel_size=7, strides=[1, 1, 1, 1], padding=3, name="gen_out"))
            return x

    def discriminator(self, x, nd, reuse=False):
        with tf.variable_scope("discriminator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            size = x.get_shape()[1].value

            x = lrelu(conv2d(x, 64, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv1"))

            x = lrelu(conv2d(x, 128, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv2"))
            x = lrelu(conv2d(x, 256, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv3"))
            x = lrelu(conv2d(x, 512, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv4"))
            x = lrelu(conv2d(x, 1024, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv5"))
            x = lrelu(conv2d(x, 2048, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv6"))

            d_src = conv2d(x, 1, kernel_size=3, strides=[1, 1, 1, 1], padding=1, name="dis_src")
            d_cls = conv2d(x, nd, kernel_size=size / 64, strides=[1, 1, 1, 1], name="dis_cls")

            return tf.squeeze(d_src), tf.squeeze(d_cls)

    def build_model(self):
        # Model
        self.fake_labels = tf.random_shuffle(self.real_labels)
        self.fake_image = self.generator(self.x, self.fake_labels)
        self.recon_image = self.generator(self.fake_image, self.real_labels, reuse=True)
        self.real_src, self.real_cls = self.discriminator(self.x, self.c_dim)
        self.fake_src, self.fake_cls = self.discriminator(self.fake_image, self.c_dim, reuse=True)
        self.alpha = tf.random_uniform([tf.shape(self.x)[0], 1, 1, 1], 0.0, 1.0)
        self.alpha = tf.tile(self.alpha, [1, tf.shape(self.x)[1], tf.shape(self.x)[2], tf.shape(self.x)[3]])
        self.interpolated = tf.multiply(self.alpha, self.x) + tf.multiply(1 - self.alpha, self.fake_image)
        self.int_disc, self.int_cls = self.discriminator(self.interpolated, self.c_dim, True)

        # Discriminator Losses
        self.d_real_loss = - tf.reduce_mean(self.real_src)
        self.d_loss_cls = tf.keras.backend.binary_crossentropy(self.real_labels, self.real_cls, from_logits=True) / self.batch_size
        self.d_fake_loss = tf.reduce_mean(self.fake_src)
        self.d_loss = self.d_real_loss + self.d_fake_loss + self.lambda_cls * self.d_loss_cls

        # Gradient Penalty
        grads = tf.gradients(self.int_disc, [self.interpolated])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, ]))
        grad_penalty = tf.reduce_mean(tf.square(slopes - 1.))
        self.grad_loss = self.lambda_gp * grad_penalty

        # Generator Losses
        self.g_fake_loss = - tf.reduce_mean(self.fake_src)
        self.g_recon_loss = tf.reduce_mean(tf.abs(self.x - self.recon_image))
        self.g_loss_cls = tf.keras.backend.binary_crossentropy(self.fake_labels, self.fake_cls, from_logits=True) / self.batch_size
        self.g_loss = self.g_fake_loss + self.lambda_rec * self.g_recon_loss + self.lambda_cls * self.g_loss_cls

        # Optimizer
        disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        self.g_optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
        self.d_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
        self.d_gp_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
        self.disc_grads_and_vars = self.d_optim.compute_gradients(self.d_loss, var_list=disc_vars)
        self.disc_step = self.d_optim.apply_gradients(self.disc_grads_and_vars)
        self.gen_grads_and_vars = self.g_optim.compute_gradients(self.g_loss, var_list=gen_vars)
        self.gen_step = self.g_optim.apply_gradients(self.gen_grads_and_vars)
        self.disc_gp_grads_and_vars = self.d_gp_optim.compute_gradients(self.grad_loss, var_list=disc_vars)
        self.disc_gp_step = self.d_gp_optim.apply_gradients(self.disc_gp_grads_and_vars)

    def make_celebA_labels(self, label):
        """Generate domain labels for CelebA for debugging/testing.
        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        batch_size = self.batch_size
        black = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant([1, 0, 0], dtype=tf.float32), 0), 0),
                        [batch_size, 1, 1])  # black hair
        blond = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant([0, 1, 0], dtype=tf.float32), 0), 0),
                        [batch_size, 1, 1])  # blond hair
        brown = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant([0, 0, 1], dtype=tf.float32), 0), 0),
                        [batch_size, 1, 1])  # brown hair

        fixed_c = tf.expand_dims(label, 1)

        # single attribute transfer
        black = tf.concat([black, fixed_c[:, :, 3:]], 2)
        blond = tf.concat([blond, fixed_c[:, :, 3:]], 2)
        brown = tf.concat([brown, fixed_c[:, :, 3:]], 2)
        rev = tf.cast(tf.logical_not(tf.cast(fixed_c, dtype=bool)), dtype=tf.float32)
        age = tf.concat([fixed_c[:, :, :3], rev[:, :, 3:4], fixed_c[:, :, 4:]], 2)
        gender = tf.concat([fixed_c[:, :, :4], rev[:, :, 4:]], 2)
        fixed_c_list = tf.concat([black, blond, brown, age, gender], 1)

        # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
        # TODO
        #         if self.dataset == 'CelebA':
        #             for i in range(4):
        #                 fixed_c = label
        #                 for c in fixed_c:
        #                     if i in [0, 1, 3]:   # Hair color to brown
        #                         c[:3] = y[2]
        #                     if i in [0, 2, 3]:   # Gender
        #                         c[3] = 0 if c[3] == 1 else 1
        #                     if i in [1, 2, 3]:   # Aged
        #                         c[4] = 0 if c[4] == 1 else 1
        #                 fixed_c_list.append(fixed_c)
        return fixed_c_list

    def train(self):
        pass
