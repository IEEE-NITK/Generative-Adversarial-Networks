import tensorflow as tf
import random


class DataLoader:
    def __init__(self, sess, image_path, metadeta_path, crop_size, image_size, batch_size, mode):
        # load data here
        self.mode = mode
        self.sess = sess
        self.image_dir = image_path
        self.lines = open(metadeta_path, "r").readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}

        self.image_size = image_size
        self.crop_size = crop_size
        self.batch_size = batch_size

    def load_dataset(self):
        imagepaths, labels = list(), list()
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.idx2attr[i] = attr
            self.attr2idx[attr] = i

        self.selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[2:]
        random.shuffle(lines)
        for i, line in enumerate(lines):
            splits = line.split()
            filename = './data/celebA/images/' + splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)
            if i in range(5):
                print(label, filename)
            imagepaths.append(filename)
            labels.append(label)

        # Convert to Tensor
        imagepath_placeholder = tf.placeholder(tf.string, len(imagepaths))
        label_placeholder = tf.placeholder(tf.float32, [len(labels), 5])

        dataset = tf.data.Dataset.from_tensor_slices((imagepath_placeholder, label_placeholder))
        dataset = dataset.map(self.process_data)
        dataset = dataset.batch(self.batch_size)

        iterator = dataset.make_initializable_iterator()

        self.sess.run(iterator.initializer, feed_dict= {
            imagepath_placeholder: imagepaths,
            label_placeholder: labels
        })
        return iterator

    def process_data(self, imagepath, label):
        image = tf.read_file(imagepath)
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, self.crop_size, self.crop_size)
        image = tf.image.resize_images(image, [self.image_size, self.image_size])  # bilinear
        image.set_shape([self.image_size, self.image_size, 3])
        image = tf.div(image, tf.constant(255.0))
        # image *= 2
        # image -= 1
        return image, label