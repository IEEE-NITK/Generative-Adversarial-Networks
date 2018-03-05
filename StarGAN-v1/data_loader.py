import numpy as np
import tensorflow as tf
import random


class DataLoader:

    def __init__(self, image_path, metadeta_path, crop_size, image_size, batch_size, mode):
        # load data here
        self.mode = mode
        self.image_dir = image_path
        self.lines = open(metadeta_path, "r").readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}

        self.image_size = image_size
        self.crop_size = crop_size
        self.batch_size = batch_size

        random.seed(7)
        # self.preprocess()

        # if mode == "train":
        #     self.num_data = len(self.train_filenames)
        #     print(self.num_data)

        #     self.path_queue = tf.train.string_input_producer(self.train_filenames, shuffle=True, seed=123)
        #     self.label_queue = tf.train.input_producer(np.array(self.train_labels, dtype=np.float32), shuffle=True, seed=123)
        #     reader = tf.WholeFileReader()
        #     paths, contents = reader.read(self.path_queue)
        #     label = self.label_queue.dequeue()
        #     raw_input = tf.image.decode_jpeg(contents, channels=3)
        #     raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        #     # process image
        #     raw_input = tf.image.resize_image_with_crop_or_pad(raw_input, crop_size, crop_size)
        #     raw_input = tf.image.resize_images(raw_input, [image_size, image_size])  # bilinear
        #     raw_input.set_shape([image_size, image_size, 3])
        #     # raw_input=tf.image.random_flip_left_right(raw_input)
        #     raw_input = raw_input * 2
        #     raw_input = raw_input - 1

        # elif mode == "test":
        #     self.num_data = len(self.test_filenames)

        #     path_queue = tf.train.string_input_producer(self.test_filenames, shuffle=True, seed=134)
        #     label_queue = tf.train.input_producer(self.test_labels, shuffle=True, seed=134)
        #     reader = tf.WholeFileReader()
        #     paths, contents = reader.read(path_queue)
        #     label = label_queue.dequeue()
        #     raw_input = tf.image.decode_jpeg(contents)
        #     raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        #     # process image
        #     raw_input = tf.image.resize_image_with_crop_or_pad(raw_input, crop_size, crop_size)
        #     raw_input = tf.image.resize_images(raw_input, [image_size, image_size])  # bilinear
        #     raw_input.set_shape([image_size, image_size, 3])
        #     # raw_input = tf.image.random_flip_left_right(raw_input)
        #     raw_input = raw_input * 2
        #     raw_input = raw_input - 1

        # # Batch
        # self.batch = tf.train.shuffle_batch([raw_input, label], batch_size=batch_size, num_threads=4, capacity=50000,
        #                                     min_after_dequeue=10)
    
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
        imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        # Build a TF Queue, shuffle data
        image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                     shuffle=True)

        # Read images from disk
        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=3)

        # Resize images to a common size
        image = tf.image.resize_image_with_crop_or_pad(image, self.crop_size, self.crop_size)
        image = tf.image.resize_images(image, [self.image_size, self.image_size])  # bilinear
        image.set_shape([self.image_size, self.image_size, 3])
        # Normalize
        image = image * 1.0/127.5 - 1.0

        # Create batches
        X, Y = tf.train.batch([image, label], batch_size=self.batch_size,
                              capacity=self.batch_size * 8,
                              num_threads=4)
        
        return X, Y

    def preprocess(self):
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
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)
                        

            if i < 1999:
                self.test_filenames.append(self.image_dir + filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(self.image_dir + filename)
                self.train_labels.append(label)
    