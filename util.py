"""Utilities
"""
import re
import base64

import numpy as np

from PIL import Image
import io
from io import BytesIO
import base64

import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import webcolors


## remove cache 
from functools import wraps, update_wrapper
from datetime import datetime
from flask import make_response

# def nocache(view):
#   @wraps(view)
#   def no_cache(*args, **kwargs):
#     response = make_response(view(*args, **kwargs))
#     response.headers['Last-Modified'] = datetime.now()
#     response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '-1'
#     return response      
#   return update_wrapper(no_cache, view)
###############


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig,ax

    

# def np_to_64(img):

#     img = Image.fromarray(img.astype("uint8"))
#     rawBytes = io.BytesIO()
#     img.save(rawBytes, "PNG")
#     rawBytes.seek(0)
#     img_base64 = base64.b64encode(rawBytes.read())

#     return img_base64

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

def count_color(masked_image):
    '''img:(n,3) array'''
    color_count={}
    for i in masked_image:
        rgb_tuple=tuple(i)
        if rgb_tuple not in color_count.keys():
            color_count[rgb_tuple]=1
        else:
            color_count[rgb_tuple]+=1

    return color_count

def count_color_dom(count_dict):
    most_color=sorted(count_dict.items(),key=(lambda x:x[1]),reverse=True)[0][0]
    return most_color


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def extract_color(image,mask):
    '''mask:(h,w,n), return :np.array[[r1,b1,g1],[r2,b2,g2],...]'''
    
    num=mask.shape[2]
    colors=[]
    
    for i in range(num):
        single_mask=mask[:,:,i]
        
        dom_color=main_color(image,single_mask)
        colors.append(dom_color)
        
    return np.array(colors)
    
def main_color(image,single_mask):
    
    cluster=10
    
    masked_image=image[single_mask]

    dec=DominantColors(masked_image,cluster)
    dom_color=dec.dominantColors()[0]
 
    return dom_color



class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self,image,clusters,roi=None,):
        '''roi:np.array([x1,y1,x2,y2])'''
        self.CLUSTERS = clusters
        self.IMAGE = image
        self.roi=roi
        
    def dominantColors(self):
    
        #read image
        
        #img=self.IMAGE[self.roi[0]:self.roi[2],self.roi[1]:self.roi[3],:]
        
        #img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        #self.IMAGE = img
        
  
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(self.IMAGE)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)
    
    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def plotClusters(self):
        #plotting 
        fig = plt.figure()
        ax = Axes3D(fig)        
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color = self.rgb_to_hex(self.COLORS[label]))
        plt.show()
        
        
    def plotHistogram(self):
       
        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
       
        #create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        
        #appending frequencies to cluster centers
        colors = self.COLORS
        
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 
        
        #creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0
        
        #creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500
            
            #getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]
            
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
            start = end	
        
        #display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()
