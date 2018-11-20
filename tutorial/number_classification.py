#from https://www.tensorflow.org/tutorials/keras/basic_classification

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Helper libraries
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2

from sklearn.model_selection import train_test_split


from tutorial.my_utils import *

#fashion_mnist = keras.datasets.fashion_mnist

class_names = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine']


(train_images, train_labels), (test_images, test_labels) = load_data('/opt/tmp/out')
class_names = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nice']


print('train images: ' , train_images.shape)
print('labels :',  train_labels)
print('test images: ' , test_images.shape)
print('labels : ', test_labels)

train_images = train_images / 255.0

test_images = test_images / 255.0

#change 0 to 1 to check image
if(0) :
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(10,10))
    for i in range(36):
        plt.subplot(6,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        #plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i]])

    plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(36, 24)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=500)

# serialize model to JSON
model_json = model.to_json()
with open("/opt/tmp/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/opt/tmp/model.h5")
print("Saved model to disk")


if(0):
    plt.figure(figsize=(10,10))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)

        #now found the number and label it
        image_height =36
        image_width = 24
        x = img_to_array(test_images[i])
        x = x.reshape(1, image_height, image_width)
        number = model.predict_classes(x)
        number1 = model.predict_proba(x)
        plt.xlabel(number)

    plt.show()

#todo slide over a whole image and try to find the number
#first try to found 'five'
fullimagefile = '/opt/tmp/image/verysmall.jpg'
#fullimagefile = '/opt/tmp/image/small.jpg'
image = cv2.imread(fullimagefile)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#get a list of small images sliding over the image
#we try 5 first (which we cut with width = 22, heigth = 32)

sliding_boxes, positions = sliding_over(image, 22, 32, 3, 3)
print('number of boxes = ', len(sliding_boxes))
probs = []
detected_boxes = []
i = 0;
for box in sliding_boxes:
    i = i + 1
    x = img_to_array(box)
    image_height =36
    image_width = 24
    x = x.reshape(1, image_height, image_width)
    prob = model.predict_proba(x)
    probs.append(prob[0][5])
    if(prob[0][5]  > 0.9):
        print(prob[0])
        detected_boxes.append(positions[i-1])
        #plt.figure()
        #plt.imshow(box)
        #plt.show()

print("detected : " , detected_boxes)
if(1):
    # Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    # Create a Rectangle patch
    index = 0
    for (x0, y0, x1, y1) in detected_boxes:
        print((x0, y0, x1, y1) )
        rect = patches.Rectangle((x0,y0),x1,y1,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        index = index + 1
        if(index > 10):
            break
    plt.show()







