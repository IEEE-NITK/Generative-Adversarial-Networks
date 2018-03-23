from base.base_model import BaseModel
from utils.generator import generator
from utils.discriminator import discriminator
import tensorflow as tf


class StarGAN(BaseModel):
    def __init__(self, config):
        super(StarGAN, self).__init__(config)

        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.image_size = config.image_size

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

        self.build_model()
        self.init_saver()

    def build_model(self):
        self.real_x = tf.placeholder(tf.float32, [None, self.image_size, self.image_size,
                                                  3], name='real_x')
        self.real_label = tf.placeholder(tf.float32, [None, self.c_dim], name='real_label')
        self.fake_label = tf.placeholder(tf.float32, [None, self.c_dim], name='fake_label')

        self.epsilon = tf.placeholder(tf.float32, [None, 1, 1, 1], name='epsilon')
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

        self.fake_x = generator(self.real_x, self.fake_label)
        self.rec_x = generator(self.fake_x, self.real_label, reuse=True)

        self.fake_x_src, self.fake_x_cls = discriminator(self.fake_x, self.c_dim)
        self.real_x_src, self.real_x_cls = discriminator(self.real_x, self.c_dim, reuse=True)
        

    def init_saver(self):
        self.saver = tf.train.Saver()
