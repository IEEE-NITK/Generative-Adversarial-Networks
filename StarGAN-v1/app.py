from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
import os
import io as imio
import numpy as np
import base64
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.misc import toimage
from io import StringIO
import re
from PIL import Image
import tensorflow as tf
from model import StarGAN
from utils.utils import *
from data_loader import DataLoader


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
config = process_config("stargan_config.json")
sess = tf.InteractiveSession()
data = DataLoader(sess, "./data/celebA/images/", "./data/celebA/list_attr_celeba.txt", 178, 128, 16, "train")
stargan = StarGAN(sess, config, data)
sess.run(tf.global_variables_initializer())

could_load, checkpoint_counter, checkpoint_epoch = stargan.load(stargan.model_dir)
if could_load:
    print("Successfully loaded checkpoint")
else:
    print("Failed to load checkpoint")

def process_label(l):
    label = np.zeros((1,5))
    label[0][int(l)] = 0
    label[0][3] = 0
    label[0][4] = 0
    # image = np.asarray(bytearray(image.read()), dtype="float32")
    # h = np.sqrt(image.shape[0] // 3)
    # print(h)
    # print()
    # arr = np.asarray(image.read(), dtype="float32")
    return label

i = 0

def serve_pil_image(pil_img):
    img_io = imio.BytesIO()
    pil_img.save(img_io, 'PNG', quality=80)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/', methods=['GET'])
def index():
    return "Welcome to STARGAN"

@app.route('/translate', methods=['POST'])
@cross_origin()
def translate_image():
    file = request.files['image']
    # image_str = request.form['image_str']
    # image_str = base64.decodestring(image_str)
    # image_str = image_str[image_str.find(",")+1:]
#     image_data = re.sub('^data:image/.+;base64,', '', request.form['data'])
#     im = Image.open(imio.BytesIO(base64.b64decode(image_data)))
    file.save(app.config['UPLOAD_FOLDER'] + '/sample'+ str(i) + '.png')
    labels = request.form['label']
    l = process_label(labels)
#     im.write(image_data)
#     im.close()
    # print(file.read())
    sample = sess.run(stargan.gen_sample, feed_dict={
        stargan.im_in_label: l,
        stargan.im_in: [app.config['UPLOAD_FOLDER'] + '/sample'+ str(i) + '.png']
    })
    os.remove(app.config['UPLOAD_FOLDER'] + '/sample'+ str(i) +'.png')
    print(sample.shape)
    return serve_pil_image(toimage(sample[0].reshape([128, 128, 3])))