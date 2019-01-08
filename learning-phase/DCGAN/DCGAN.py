import os
import time
import pandas as pd

from helpers import *


class DCGAN(object):
    def __init__(self, sess, input_height=28, input_width=28, crop=True,
                 batch_size=64, sample_num=64, output_height=28, output_width=28,
                 y_dim=10, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024,
                 checkpoint_dir='checkpoints', dataset='mnist'):
        self.sess = sess
        self.crop = crop

        self.dataset = dataset

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.checkpoint_dir = checkpoint_dir

        if dataset == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.data_X = np.reshape(self.data_X, [-1, self.input_height, self.input_width, 1])
        elif dataset == 'cifar':
            self.data_X, self.data_y = self.load_cifar()
            self.data_X = np.reshape(self.data_X, [-1, 3,self.input_height, self.input_width])
            self.data_X = np.rollaxis(self.data_X, 1, 4)

        self.c_dim = self.data_X[0].shape[-1]

        self.build_model()

    def build_model(self):
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_inputs = self.data_X[0:self.sample_num]
        sample_labels = self.data_y[0:self.sample_num]

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print("Successfully loaded checkpoint")
        else:
            print("Failed to load checkpoint")

        for epoch in range(25):
            batch_idxs = self.data_y.shape[0] // self.batch_size
            for idx in range(batch_idxs):
                batch_images = self.data_X[idx * 64:(idx + 1) * 64]
                batch_labels = self.data_y[idx * 64:(idx + 1) * 64]

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                self.sess.run([d_optim], feed_dict={self.inputs: batch_images,
                                                    self.z: batch_z,
                                                    self.y: batch_labels})

                self.sess.run([g_optim], feed_dict={self.z: batch_z, self.y: batch_labels})

                self.sess.run([g_optim], feed_dict={self.z: batch_z, self.y: batch_labels})

                errD = self.d_loss.eval({
                    self.z: batch_z,
                    self.inputs: batch_images,
                    self.y: batch_labels
                })
                errG = self.g_loss.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD, errG))

                if np.mod(counter, 50) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_z,
                            self.inputs: sample_inputs,
                            self.y: sample_labels,
                        }
                    )
                    save_images(samples, image_manifold_size(samples.shape[0]),
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 100) == 1:
                    self.save(self.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            if self.dataset == 'mnist':
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = leaky_relu(conv_layer(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = leaky_relu(self.d_bn1(conv_layer(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = concat([h1, y], 1)

                h2 = leaky_relu(self.d_bn2(dense(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = dense(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3
            elif self.dataset == 'cifar':
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = leaky_relu(conv_layer(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = leaky_relu(self.d_bn1(conv_layer(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                # h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = conv_cond_concat(h1, yb)
                # h1 = concat([h1, y], 1)

                h2 = leaky_relu(self.d_bn2(conv_layer(h1, self.df_dim * 2 + self.y_dim, name='d_h2_conv')))
                h2 = tf.reshape(h2, [self.batch_size, -1])
                h2 = concat([h2, y], 1)

                h3 = leaky_relu(self.d_bn3(dense(h2, self.dfc_dim * 2, 'd_h3_lin')))
                h3 = concat([h3, y], 1)

                h4 = dense(h3, 1, 'd_h4_lin')

                return tf.nn.sigmoid(h4), h4

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:

            if self.dataset == 'mnist':
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(dense(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(dense(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(conv_transpose_layer(h1,
                                                    [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(conv_transpose_layer(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

            elif self.dataset == 'cifar':
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4, s_h8, s_h16 = int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16)
                s_w2, s_w4, s_w8, s_w16 = int(s_w / 2), int(s_w / 4), int(s_w / 8), int(s_w / 16)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(dense(z, self.gf_dim * 4 * s_h16 * s_w16, 'g_h0_lin')))
                h0 = tf.reshape(h0, [self.batch_size, s_h16, s_w16, self.gf_dim * 4])
                h0 = conv_cond_concat(h0, yb)

                h1 = tf.nn.relu(self.g_bn1(conv_transpose_layer(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], 'g_h1_lin')))

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(conv_transpose_layer(h1,
                                                                [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                                                                name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                h3 = tf.nn.relu(self.g_bn3(conv_transpose_layer(h2, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h3')))

                h3 = conv_cond_concat(h3, yb)

                return tf.nn.tanh(conv_transpose_layer(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4'))

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            if self.dataset == 'mnist':
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(dense(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    dense(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    conv_transpose_layer(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(conv_transpose_layer(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

            elif self.dataset == 'cifar':
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4, s_h8, s_h16 = int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16)
                s_w2, s_w4, s_w8, s_w16 = int(s_w / 2), int(s_w / 4), int(s_w / 8), int(s_w / 16)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(dense(z, self.gf_dim * 4 * s_h16 * s_w16, 'g_h0_lin')))
                h0 = tf.reshape(h0, [self.batch_size, s_h16, s_w16, self.gf_dim * 4])
                h0 = conv_cond_concat(h0, yb)

                h1 = tf.nn.relu(self.g_bn1(conv_transpose_layer(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], 'g_h1_lin')))

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(conv_transpose_layer(h1,
                                                                [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                                                                name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                h3 = tf.nn.relu(
                    self.g_bn3(conv_transpose_layer(h2, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h3')))

                h3 = conv_cond_concat(h3, yb)

                return tf.nn.tanh(conv_transpose_layer(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4'))

    def load_mnist(self):
        data = pd.read_csv('train.csv')
        # data = data.drop('label', axis=1)
        X = data.iloc[:, 1:].values
        X = X.astype(np.float)

        # train_df = np.multiply(train_df, 1.0 / 255.0)

        labels_flat = data.iloc[:, 0].values.ravel()

        def dense_to_one_hot(labels_dense, num_classes):
            num_labels = labels_dense.shape[0]
            index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
            return labels_one_hot

        y = dense_to_one_hot(labels_flat, self.y_dim)
        y = y.astype(np.uint8)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        return X / 255., y

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_cifar(self):
        tr1 = self.unpickle('data/data_batch_1')
        tr2 = self.unpickle('data/data_batch_2')
        tr3 = self.unpickle('data/data_batch_3')
        tr4 = self.unpickle('data/data_batch_4')
        tr5 = self.unpickle('data/data_batch_5')
        te = self.unpickle('data/test_batch')

        X = np.concatenate((tr1[b'data'], tr2[b'data'], tr3[b'data'],
                            tr4[b'data'], tr5[b'data'], te[b'data']), axis=0)
        y = np.concatenate((tr1[b'labels'], tr2[b'labels'], tr3[b'labels'],
                            tr4[b'labels'], tr5[b'labels'], te[b'labels']), axis=0)

        def dense_to_one_hot(labels_dense, num_classes):
            num_labels = labels_dense.shape[0]
            index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
            return labels_one_hot

        y = dense_to_one_hot(y, self.y_dim)
        y = y.astype(np.uint8)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        return X / 255., y

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            'default', self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print("Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print("Successfully read {}".format(ckpt_name))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0
