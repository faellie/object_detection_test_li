#from https://www.tensorflow.org/tutorials/keras/basic_classification

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Helper libraries
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import errno

import cv2
#import glob
#import scipy
from sklearn.model_selection import train_test_split

print(tf.__version__)
#fashion_mnist = keras.datasets.fashion_mnist

class_names = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine', 'bg']

#slid over the image and create a list boxes of width and height
# stride_x and stride_y define the slid pixels in x and y dir
def sliding_over(image, box_width=24, box_height=36, stride_x= 2, stride_y = 2, save_to_file=''):
    boxes = []
    postions=[]
    imgheight, imgwidth  = image.shape[:2]
    x_slids = (imgwidth - box_width)//stride_x
    y_slids = (imgheight-box_height)//stride_y
    for x0 in range(0, imgwidth, stride_x):
        for y0 in  range(0, imgheight, stride_y):
            x1 = x0   + box_width
            y1 = y0 + box_height
            if(x1 <= imgwidth and y1 <= imgheight):
                box = (x0, y0, x1, y1)
                #print(box)
                boxImage = image[y0:y1, x0:x1]
                #plt.figure()
                #plt.imshow(boxImage)
                #plt.show()
                boxImage = cv2.resize(boxImage, (24, 36), cv2.INTER_AREA)
                boxes.append(boxImage)
                postions.append(box)
            else:
                print('skip', box)
    if save_to_file:
        make_sure_path_exists(save_to_file)
        index = 0
        for box in boxes:
            #filepath = os.path.join(save_to_file, 'box'+ str(index) + '.jpg')
            (x0, y0, x1, y1) = postions[index]
            filepath = os.path.join(save_to_file, str(x0) + '_' + str(y0) + '.jpg')
            print('writing to ', filepath)
            cv2.imwrite(filepath, box)
            index = index + 1
    return boxes, postions;





#Assume picture of each number is saved under sub dir with the 'number'
#i.e all picture of 1 is saved under a sub dir of 'one" etc.
def extractLabelFromPath(fullpathfile):
    if('/zero/' in fullpathfile):
        return 0
    if('/one/' in fullpathfile):
        return 1
    if('/two/' in fullpathfile):
        return 2
    if('/three/' in fullpathfile):
        return 3
    if('/four/' in fullpathfile):
        return 4
    if('/five/' in fullpathfile):
        return 5
    if('/six/' in fullpathfile):
        return 6
    if('/seven/' in fullpathfile):
        return 7
    if('/eight/' in fullpathfile):
        return 8
    if('/nine/' in fullpathfile):
        return 9
    if('/bg/' in fullpathfile):
        return 10
    else:
        return -1

def load_data(path):
    labels=[]
    train_files=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            fullpathfile = os.path.join(root, file)
            label = extractLabelFromPath(fullpathfile)
            if(label >= 0):
                train_files.append(fullpathfile)
                labels.append(extractLabelFromPath(fullpathfile))
    # got labels, files
    image_height =36
    image_width = 24
    channels = 1
    dataset = np.ndarray(shape=(len(train_files),  image_height, image_width),
                         dtype=np.float32)
    i = 0
    #followed https://www.kaggle.com/lgmoneda/from-image-files-to-numpy-arrays
    for _file in train_files:
        img = load_img(_file, grayscale=True)  # this is a PIL image
        #img.thumbnail((image_width, image_height))
        x = img_to_array(img)
        x = x.reshape(image_height, image_width)
        # Normalize
        x = (x - 128.0) / 128.0
        dataset[i] = x
        i += 1
        if i % 250 == 0:
            print("%d images to array" % i)

    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2)

    return (x_train, y_train), (x_test, y_test)


def load_images_for_detect(path):
    train_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            fullpathfile = os.path.join(root, file)
            train_files.append(fullpathfile)
    # got labels, files
    image_height =36
    image_width = 24
    channels = 1
    dataset = np.ndarray(shape=(len(train_files),  image_height, image_width),
                         dtype=np.float32)
    i = 0
    #followed https://www.kaggle.com/lgmoneda/from-image-files-to-numpy-arrays
    for _file in train_files:
        img = load_img(_file, grayscale=True)  # this is a PIL image
        #img.thumbnail((image_width, image_height))
        if(0):
            plt.figure()
            plt.imshow(img)
            plt.show()
        x = img_to_array(img)
        x = x.reshape(image_height, image_width)
        # Normalize
        x = (x - 128.0) / 128.0
        if(0):
            plt.figure()
            plt.imshow(x)
            plt.xlabel(_file)
            plt.show()
        dataset[i] = x
        i += 1

    return dataset

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise






