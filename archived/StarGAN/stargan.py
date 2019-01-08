import tensorflow as tf
from data_loader.data_loader import DataLoader
from models.stargan_model import StarGAN
from trainers.stargan_trainer import StarGANTrainer
from utils.config import process_config
from utils.utils import get_args
from utils.logger import Logger


def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("Missing Arguments")
        exit(0)
    sess = tf.Session()
    data = DataLoader("./data/celebA/images/", "./data/celebA/list_attr_celeba.txt", 178, 128, 16, "train")
    model = StarGAN(config, data)
    logger = Logger(sess, config)
    trainer = StarGANTrainer(sess, model, data, config, logger)
    trainer.train()


if __name__ == '__main__':
    main()
