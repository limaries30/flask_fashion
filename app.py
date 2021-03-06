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
from util import base64_to_pil,np_to_base64
from util import DominantColors,closest_colour,get_colour_name,main_color,extract_color,get_ax,count_color,count_color_dom
import io
from io import BytesIO
import base64
from PIL import Image
import webcolors



# Declare a flask app
app = Flask(__name__)


# Model saved with Keras model.save()

config = coco_main.FashionConfig()

COCO_MODEL_PATH ='./models/mask_rcnn_coco_0069.h5'  #path to model
MODEL_DIR = './models'
DEVICE = "/gpu:0" 
TEST_MODE = "inference"


config = coco_main.FashionConfig()
dataset = coco_main.FashionDataset()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BACKBONE='resnet50'

config = InferenceConfig()
config.display()

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

print("Loading weights ", COCO_MODEL_PATH)
model.load_weights(COCO_MODEL_PATH, by_name=True)    
print("Loaded")

CLASS_NAMES=['BG',\
 'Short Sleeved Shirt',\
 'Long Sleeved Shirt',\
 'Short Sleeved Outwear',\
 'Long Sleeved Outwear',\
 'Vest',\
 'Sling',\
 'Shorts',\
 'Trousers',\
 'Skirt',\
 'Short Sleeved Dress',\
 'Long Sleeved Dress',\
 'Vest Dress',\
 'Sling Dress']



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
        masked_image = img.astype(np.uint32).copy()


        '''Make prediction'''
        preds = model.detect([img], verbose=1)[0]
        print('Detection Finished')
        N = preds['rois'].shape[0]
        masks=preds['masks']
        colors = visualize.random_colors(N)

        if not N:
            print("\n*** No instances to display *** \n")
            final_sentence="인공지능도 반해버린 당신"
            return jsonify(image_response=None,result=final_sentence, probability=None)

        '''Masking'''
        for i in range(N):
            mask = masks[:, :, i]
            masked_image = visualize.apply_mask(masked_image, mask, colors[i])
        
        '''Plotting mask and bbox'''
        fig,ax = get_ax(1)
        _=visualize.display_instances(img, preds['rois'], preds['masks'], preds['class_ids'], 
                            CLASS_NAMES, preds['scores'], ax=ax,
                            title="Predictions")
        print('Masking Finished')

        '''flask img to bytes'''
        io = BytesIO()
        fig.savefig(io, format='png')
        data = base64.encodestring(io.getvalue()).decode('utf-8')
        original_img_base64=np_to_base64(img)
        masked_img_base64=np_to_base64(masked_image)
        total_img_base64='data:image/png;base64,'+data


        '''return most frequent color in [(r,g,b),...]'''
        color_count=[count_color(img[masks[:,:,i]]) for i in range(N)]
        colors_rgb=list(map(lambda x:count_color_dom(x),color_count))
        print(colors_rgb)
       
        '''return dominant color in [(r,g,b),...] using k-means'''
        #colors_rgb=extract_color(img,masks)

        '''get color name according to css3'''
        #color_names=list(map(lambda x:get_colour_name(x)[-1],colors_rgb))

        '''change rgb to hex'''
        color_names=list(map(lambda x:webcolors.rgb_to_hex(x),colors_rgb))
        '''get cloth names'''
        cloth_names=np.array(CLASS_NAMES)[preds['class_ids']].tolist()
        print(cloth_names)
        '''make dict={cloth name:color name,...'''
        fashion_info=dict(zip(cloth_names,color_names))

        # result_sentence=[]
        # for x,y in zip(reversed(color_names),reversed(clothe_names)):
        #     result_sentence.append(x+' '+y)
        # final_sentence=' and '.join(result_sentence)

  

        return jsonify(image_response=total_img_base64,result=fashion_info, probability=None)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('localhost', 8080), app)
    http_server.serve_forever()
