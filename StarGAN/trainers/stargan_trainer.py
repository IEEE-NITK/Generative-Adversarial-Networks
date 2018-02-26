from base.base_train import BaseTrain
from tqdm import tqdm
import tensorflow as tf
import numpy as np


class StarGANTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(StarGANTrainer, self).__init__(sess, model, data, config, logger)
        # Training settings
        self.dataset = config.dataset
        self.num_iters = config.num_iters
        self.batch_size = config.batch_size
        self.fixed_c_list = self.make_celebA_labels(model.real_labels)
        self.fixed_c_list = tf.unstack(self.fixed_c_list, axis=1)
        self.fixed_c_list = tf.stack(self.fixed_c_list, axis=0)

        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        disc_vars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

        self.g_step = model.g_optim.minimize(model.g_loss, var_list=gen_vars)
        self.d_step = model.d_optim.minimize(model.d_loss, var_list=disc_vars)

    def train_epoch(self):
        for step in range(self.num_iters):
            g_loss, d_loss = self.train_step()
            print("Generator Loss: {}, Discriminator Loss: {}".format(g_loss, d_loss))


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

        fixed_c_list = []
        fixed_c = tf.expand_dims(label, 1)

        # single attribute transfer
        black = tf.concat([black, fixed_c[:, :, 3:]], 2)
        blond = tf.concat([blond, fixed_c[:, :, 3:]], 2)
        brown = tf.concat([brown, fixed_c[:, :, 3:]], 2)
        rev = tf.cast(tf.logical_not(tf.cast(fixed_c, dtype=bool)), dtype=tf.float32)
        age = tf.concat([fixed_c[:, :, :3], rev[:, :, 3:4], fixed_c[:, :, 4:]], 2)
        gender = tf.concat([fixed_c[:, :, :4], rev[:, :, 4:]], 2)
        fixed_c_list = tf.concat([black, blond, brown, age, gender], 1)
        print(fixed_c_list.shape)

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

    def train_step(self):
        for _ in range(5):
            _, d_loss = self.sess.run([self.d_step, self.model.d_loss])

        _, g_loss = self.sess.run([self.g_step, self.model.g_loss])

        return g_loss, d_loss