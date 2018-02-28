import tensorflow as tf
from model import StarGAN
from utils.utils import get_args, process_config
from data_loader import DataLoader


def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("Missing Arguments")
        exit(0)
    data = DataLoader("./data/celebA/images/", "./data/celebA/list_attr_celeba.txt", 178, 128, 16, "train")
    sess = tf.Session()
    stargan = StarGAN(sess, config, data)
    stargan.train()


if __name__ == '__main__':
    main()
