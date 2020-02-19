import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import IPython.display as display
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from fashion_cnn import coco_main
from mrcnn.model import log

# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# Model saved with Keras model.save()

config = coco_main.FashionConfig()


COCO_MODEL_PATH ='./models/mask_rcnn_coco_0_87_1_0069.h5'  #path to model
MODEL_DIR = './models'
DEVICE = "/cpu:0" 
TEST_MODE = "inference"


config = coco_main.FashionConfig()
dataset = coco_main.FashionDataset()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

print("Loading weights ", COCO_MODEL_PATH)
model.load_weights(COCO_MODEL_PATH, by_name=True)    

CLASS_NAMES=['BG',\
 'short_sleeved_shirt',\
 'long_sleeved_shirt',\
 'short_sleeved_outwear',\
 'long_sleeved_outwear',\
 'vest',\
 'sling',\
 'shorts',\
 'trousers',\
 'skirt',\
 'short_sleeved_dress',\
 'long_sleeved_dress',\
 'vest_dress',\
 'sling_dress']



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
        # Get the image from post request
        img = base64_to_pil(request.json)
        img=np.array(img)
        print(img.shape)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model.detect([img], verbose=1)
        result=preds[0]['class_ids'][0]
        result=CLASS_NAMES[result]
        pred_proba='TODO'

        # Process your result for human
        # pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        # result = str(pred_class[0][0][1])               # Convert to string
        # result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('localhost', 8080), app)
    http_server.serve_forever()
