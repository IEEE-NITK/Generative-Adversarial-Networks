from base.base_model import BaseModel
import tensorflow as tf
from utils.generator import generator
from utils.discriminator import discriminator


class StarGAN(BaseModel):
    def __init__(self, config):
        super(StarGAN, self).__init__(config)

        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size

        self.use_tensorboard = config.use_tensorboard
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Model
        self.x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], "x")
        self.alpha = tf.placeholder(tf.float32, [None, 1, 1, 1], "alpha")
        self.real_labels = tf.placeholder(tf.float32, [None, self.c_dim], "real_labels")
        self.fake_labels = tf.placeholder(tf.float32, [None, self.c_dim], "fake_labels")
        self.fake_image = generator(self.x, self.fake_image)
        self.recon_image = generator(self.fake_image, self.real_labels, reuse=True)
        self.real_src, self.real_cls = discriminator(self.x, self.c_dim)
        self.fake_src, self.fake_cls = discriminator(self.fake_image, self.c_dim, reuse=True)
        self.interpolated = tf.multiply(self.alpha, self.x) + tf.multiply(1 - self.alpha, self.fake_image)
        self.int_disc, self.int_cls = discriminator(self.interpolated, self.c_dim, True)

        # Discriminator Losses
        self.d_real_loss = tf.reduce_mean(self.real_src)
        self.d_loss_cls = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.real_labels, logits=self.real_cls)
        self.d_fake_loss = tf.reduce_mean(self.fake_src)
        self.d_loss = self.d_real_loss + self.d_fake_loss + self.lambda_cls*self.d_loss_cls

        # Gradient Penalty
        grads = tf.gradients(self.int_disc, [self.interpolated])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,]))
        grad_penalty = tf.reduce_mean(tf.square(slopes - 1.))
        self.grad_loss = self.lambda_gp * grad_penalty

        # Generator Losses
        self.g_fake_loss = tf.reduce_mean(self.fake_image)
        self.g_recon_loss = tf.reduce_mean(tf.abs(self.x - self.recon_image))
        self.g_loss = self.g_fake_loss + self.lambda_rec * self.g_recon_loss

        # Optimizer
        self.g_optim = tf.train.AdamOptimizer(self.g_lr, self.beta1, self.beta2)
        self.d_optim = tf.train.AdamOptimizer(self.d_lr, self.beta1, self.beta2)



    def init_saver(self):
        # initalize the tensorflow saver that will be used in saving the checkpoints.
        pass
