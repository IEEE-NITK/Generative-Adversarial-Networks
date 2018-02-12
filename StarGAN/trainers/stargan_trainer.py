from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class StarGANTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(StarGANTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        pass

    def train_step(self):
        pass
