from base.base_model import BaseModel
import tensorflow as tf


class StarGAN(BaseModel):
    def __init__(self, config):
        super(StarGAN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

    def init_saver(self):
        # initalize the tensorflow saver that will be used in saving the checkpoints.
        pass
