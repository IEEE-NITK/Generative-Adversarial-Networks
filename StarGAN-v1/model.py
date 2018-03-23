import tensorflow as tf
import time
from tqdm import tqdm
from utils.ops import *
from utils.utils import save_images, image_manifold_size
from data_loader import DataLoader
from utils.optim import OptimisticAdam
import os


class StarGAN:
    def __init__(self, sess, config, data):

        # Model hyper-parameters
        self.sess = sess
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.max_steps = config.max_steps

        # Hyper-parameters
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # self.im_in = tf.placeholder(dtype=tf.float32, shape=[1, 128, 128, 3])
        self.im_in_label = tf.placeholder(dtype=tf.float32, shape=[1, 5])
        self.iter = data.load_dataset()
        self.x, self.real_labels = self.iter.get_next()

        # self.iter1 = data.load_dataset1()
        # self.x1, self.real_labels1 = self.iter1.get_next()
        # self.iter2 = data.load_dataset2()
        # self.x2, self.real_labels2 = self.iter2.get_next()

        self.epochs = config.epochs
        self.model_dir = config.model_dir
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.fixed_c_list = self.make_celebA_labels(self.real_labels)
        self.fixed_c_list = tf.unstack(self.fixed_c_list, axis=1)
        self.fixed_c_list = tf.stack(self.fixed_c_list, axis=0)

        self.build_model()
        self.saver = tf.train.Saver(max_to_keep=1)

    def generate_fake_labels(self):
        l1 = tf.random_shuffle(self.real_labels[:, :3])
        l2 = tf.cast(tf.random_uniform([self.batch_size, 2], 0, 2, dtype=tf.int32), dtype=tf.float32)
        return tf.concat([l1, l2], axis=1)

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
                              name="gen_res_conv{}_1".format(i))
                x = x_ + res2
                x_ = x

            x = relu(instance_norm(deconv2d(x, [16, 64, 64, 128], kernel_size=4, strides=[1, 2, 2, 1], padding=1,
                                            name="gen_us_deconv1"), name="gen_in3_1"))
            x = relu(instance_norm(deconv2d(x, [16, 128, 128, 64], kernel_size=4, strides=[1, 2, 2, 1], padding=1,
                                            name="gen_us_deconv2"), name="gen_in3_2"))

            x = tanh(conv2d(x, 3, kernel_size=7, strides=[1, 1, 1, 1], padding=3, name="gen_out"))
            return x
    
    def get_sample(self, c, reuse=False):
        with tf.variable_scope("generator"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            imagepath = './demo-server/uploads/sample.png'
            image = tf.read_file(imagepath)
            image = tf.image.decode_image(image, channels=3)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
            image = tf.image.resize_images(image, [128, 128])  # bilinear
            image.set_shape([128, 128, 3])
            image = tf.div(image, tf.constant(255.0))
            x = tf.expand_dims(image, 0)
            x = tf.tile(image, [16, 1, 1, 1])

            c = tf.expand_dims(tf.expand_dims(c, 1), 1)
            c = tf.tile(c, [16, x.get_shape()[1].value, x.get_shape()[2].value, 1])
            # x = tf.tile(x, [16, 1, 1, 1])
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
                              name="gen_res_conv{}_1".format(i))
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

            x = lrelu(conv2d(x, 64, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv1"))

            x = lrelu(conv2d(x, 128, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv2"))
            x = lrelu(conv2d(x, 256, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv3"))
            x = lrelu(conv2d(x, 512, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv4"))
            x = lrelu(conv2d(x, 1024, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv5"))
            x = lrelu(conv2d(x, 2048, kernel_size=4, strides=[1, 2, 2, 1], padding=1, name="dis_conv6"))

            d_src = conv2d(x, 1, kernel_size=3, strides=[1, 1, 1, 1], padding=1, name="dis_src")
            d_cls = conv2d(x, nd, kernel_size=2, strides=[1, 1, 1, 1], padding=0, name="dis_cls")
            return tf.squeeze(d_src), tf.squeeze(d_cls)

    def build_model(self):
        # Model
        self.fake_labels = self.generate_fake_labels()
        self.fake_image = self.generator(self.x, self.fake_labels)
        self.recon_image = self.generator(self.fake_image, self.real_labels, reuse=True)
        self.gen_sample = self.get_sample(self.im_in_label, reuse=True)
        self.real_src, self.real_cls = self.discriminator(self.x, self.c_dim)
        self.fake_src, self.fake_cls = self.discriminator(self.fake_image, self.c_dim, reuse=True)
        self.alpha = tf.random_uniform([tf.shape(self.x)[0], 1, 1, 1], 0.0, 1.0)
        self.alpha = tf.tile(self.alpha, [1, tf.shape(self.x)[1], tf.shape(self.x)[2], tf.shape(self.x)[3]])
        self.interpolated = tf.multiply(self.alpha, self.x) + tf.multiply(1 - self.alpha, self.fake_image)
        self.int_disc, self.int_cls = self.discriminator(self.interpolated, self.c_dim, True)

        # Gradient Penalty
        grads = tf.gradients(self.int_disc, [self.interpolated])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, ]))
        grad_penalty = tf.reduce_mean(tf.square(slopes - 1.))
        self.grad_loss = self.lambda_gp * grad_penalty

        # Discriminator Losses
        self.d_real_loss = - tf.reduce_mean(self.real_src)
        self.d_loss_cls = tf.keras.backend.binary_crossentropy(self.real_labels, self.real_cls,
                                                               from_logits=True) / self.batch_size
        self.d_fake_loss = tf.reduce_mean(self.fake_src)
        self.d_loss = self.d_real_loss + self.d_fake_loss + self.lambda_cls * self.d_loss_cls
        self.d_loss = tf.reduce_mean(self.d_loss) + self.grad_loss

        # Generator Losses
        self.g_fake_loss = - tf.reduce_mean(self.fake_src)
        self.g_recon_loss = tf.reduce_mean(tf.abs(self.x - self.recon_image))
        self.g_loss_cls = tf.keras.backend.binary_crossentropy(self.fake_labels, self.fake_cls,
                                                               from_logits=True) / self.batch_size
        self.g_loss = self.g_fake_loss + self.lambda_rec * self.g_recon_loss + self.lambda_cls * self.g_loss_cls
        self.g_loss = tf.reduce_mean(self.g_loss)

        # Optimizer
        disc_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        self.g_optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2)
        self.d_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
        # self.d_gp_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2)
        # self.disc_grads_and_vars = self.d_optim.compute_gradients(self.d_loss, var_list=disc_vars)
        self.disc_step = self.d_optim.minimize(self.d_loss, var_list=disc_vars)
        # self.gen_grads_and_vars = self.g_optim.compute_gradients(self.g_loss, var_list=gen_vars)
        self.gen_step = self.g_optim.minimize(self.g_loss, var_list=gen_vars)
        # self.disc_gp_grads_and_vars = self.d_gp_optim.compute_gradients(self.grad_loss, var_list=disc_vars)
        # self.disc_gp_step = self.d_gp_optim.apply_gradients(self.disc_gp_grads_and_vars)

    # def build_model_multi(self):
    #     self.fake_labels1 = tf.random_shuffle(self.real_labels1)
    #     self.fake_labels2 = tf.random_shuffle(self.real_labels2)

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
        return fixed_c_list

    @property
    def model_dir_(self):
        return "{}_{}_{}_{}".format(
            'default', self.batch_size,
            self.image_size, self.image_size)

    def save(self, checkpoint_dir, step, epoch):
        model_name = "StarGAN.model_{:02d}".format(epoch)
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir_)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print("Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir_)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            m = ckpt_name.find("model")
            epoch = int(ckpt_name[m+6: m+8])
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print("Successfully read {}".format(ckpt_name))
            return True, counter, epoch
        else:
            return False, 0, 0

    def train(self, mode='train'):
        print('Beginning Training: ')
        # with self.sess as sess:
        sess = self.sess
        tf.global_variables_initializer().run(session=self.sess)

        if mode == 'test' or mode == 'validation':
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            # print(checkpoint)
            self.saver.restore(sess, checkpoint)
        else:
            # checkpoint = tf.train.latest_checkpoint(self.model_dir)
            # if checkpoint:
            #     self.saver.restore(sess, checkpoint)
            #     print("Restored from checkpoint")
            counter = 0
            ep = 0
            could_load, checkpoint_counter, checkpoint_epoch = self.load(self.model_dir)
            if could_load:
                ep = checkpoint_epoch
                counter = checkpoint_counter
                print("Successfully loaded checkpoint")
            else:
                print("Failed to load checkpoint")
            start_time = time.time()
            labels = open("samples/labels.txt", "a")
            for epoch in tqdm(range(ep, self.epochs)):
                for step in tqdm(range(counter, self.max_steps)):
                    if step - counter in range(2101, 2116):
                        x, _ = sess.run([self.x, self.real_labels])
                        print(x.shape)
                        continue
                    for _ in range(5):
                        # self.x, self.real_labels = self.iter.get_next()
                        _, disc_loss, x = self.sess.run([self.disc_step, self.d_loss, self.x])
                        if step - counter > 2110:
                            print(x.shape)
                        # _ = self.sess.run([self.disc_gp_step])

                    # self.x, self.real_labels = self.iter.get_next()
                    _, gen_loss = self.sess.run([self.gen_step, self.g_loss])

                    if step % 100 == 0:
                        print("Time: {:.4f}, Epoch: {}, Step: {}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}"
                              .format(time.time() - start_time, epoch, step, gen_loss, disc_loss))
                        fake_im, real_im, fake_l, real_l = sess.run([self.fake_image, self.x, self.fake_labels, self.real_labels])
                        save_images(fake_im, image_manifold_size(fake_im.shape[0]),
                                    './samples/train_{:02d}_{:06d}.png'.format(epoch, step))
                        save_images(real_im, image_manifold_size(real_im.shape[0]),
                                    './samples/train_{:02d}_{:06d}_real.png'.format(epoch, step))
                        labels.write("{:02d}_{:06d}:\nReal Labels -\n{}\nFake Labels -\n{}\n".format(epoch, step, str(real_l), str(fake_l)))
                        print('Translated images and saved..!')

                    if step % 200 == 0:
                        self.save(self.model_dir, step, epoch)
                        print("Checkpoint saved")
                counter = 0
