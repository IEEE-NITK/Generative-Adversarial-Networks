from flask import Flask, request, send_file
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

app = Flask(__name__)


UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def preprocess(filename):
    # image = np.asarray(bytearray(image.read()), dtype="float32")
    # h = np.sqrt(image.shape[0] // 3)
    # print(h)
    # print()
    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    sample = io.imread(filename)
    sample = resize(sample, [128, 128, 3])
    plt.imshow(sample)
    plt.show()
    print(sample.shape)
    # arr = np.asarray(image.read(), dtype="float32")
    return sample

def serve_pil_image(pil_img):
    img_io = StringIO()
    pil_img.save(img_io, 'PNG', quality=80)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/translate', methods=['POST'])
def translate_image():
    # file = request.files['image']
    # image_str = request.form['image_str']
    # image_str = base64.decodestring(image_str)
    # image_str = image_str[image_str.find(",")+1:]
    image_data = re.sub('^data:image/.+;base64,', '', request.form['data'])
    im = Image.open(imio.BytesIO(base64.b64decode(image_data)))
    im = open(app.config['UPLOAD_FOLDER']+'/sample.png', 'wb')
    im.write(image_data)
    im.close()
    # print(file.read())
    file = preprocess('sample.png')
    return serve_pil_image(toimage(file))