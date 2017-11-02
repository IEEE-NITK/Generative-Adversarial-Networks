# Importing data and TensorFlow library.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Storing MNIST data in an object.

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# GLOBAL VARIABLES SECTION
# Number of pixels in length as well as breadth of each image.
image_pixels = 28

# Kernel size (dimension length of square kernel matrix)
kernel = 5


def weight_initialise(shape):
    value = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(value)


def bias_initialise(shape):
    value = tf.constant(0.1, shape=shape)
    return tf.Variable(value)


# Classifier function that takes input as tensor x. It is responsible for building the computational graph.

def classifier(x):
    # Using auto-expand to convert input tensor x to a 3-D tensor.
    input_layer = tf.reshape(x, [-1, image_pixels, image_pixels, 1])

    # Creating parameters matrix for convolution-1. Note that 32 is the number of units and hence we need 32 rows of
    # kernel x kernel matrices
    params_conv1 = weight_initialise([kernel, kernel, 1, 32])

    # Creating bias tensor for first convolution.
    bias_conv1 = bias_initialise([32])

    # Detector stage for first convolution.
    output_conv1 = tf.nn.relu(tf.nn.conv2d(input_layer, params_conv1, strides=[1, 1, 1, 1], padding='SAME')+bias_conv1)

    # Pooling stage for first convolution.
    pool_conv1 = tf.nn.max_pool(output_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Creating parameters and bias for second convolution.
    params_conv2 = weight_initialise([kernel, kernel, 32, 64])
    bias_conv2 = bias_initialise([64])

    # Detector stage for second convolution
    output_conv2 = tf.nn.relu(tf.nn.conv2d(pool_conv1, params_conv2, strides=[1, 1, 1, 1], padding='SAME')+bias_conv2)

    # Pooling stage for second convolution
    pool_conv2 = tf.nn.max_pool(output_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Producing output in fully connected layer 1.
    params_fc1 = weight_initialise([7*7*64, 1024])
    bias_fc1 = bias_initialise([1024])
    pool_conv2_flat = tf.reshape(pool_conv2, [-1, 7*7*64])
    output_fc1 = tf.nn.relu(tf.matmul(pool_conv2_flat, params_fc1) + bias_fc1)

    # The dropout probability needs to be calculated to prevent an overfit/ adaptation of parameters.
    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(output_fc1, keep_prob)
    # Now, we need to map these 1024 hidden dense units to 10 final output units

    params_fc2 = weight_initialise([1024, 10])
    bias_fc2 = bias_initialise([10])
    y_conv = tf.matmul(dropout, params_fc2) + bias_fc2

    return y_conv, keep_prob


# Declaring variables

x = tf.placeholder(tf.float32, [None, 784])
targets = tf.placeholder(tf.float32, [None, 10])
y_conv, keep_prob = classifier(x)

# Calculating loss

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(targets, 1))
correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(10000):
    print "Completed step %d" %(i)
    batch_x, batch_y = mnist.train.next_batch(50)
    sess.run(train, feed_dict={x: batch_x, targets: batch_y, keep_prob: 0.5})

print('test accuracy %g' % accuracy.eval(feed_dict={
    x: mnist.test.images, targets: mnist.test.labels, keep_prob: 1.0}))








