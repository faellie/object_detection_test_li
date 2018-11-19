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
#import glob
#import scipy
from sklearn.model_selection import train_test_split

print(tf.__version__)
#fashion_mnist = keras.datasets.fashion_mnist

class_names = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine']

#slid over the image and create a list boxes of width and height
# stride_x and stride_y define the slid pixels in x and y dir
def sliding_over(image, box_width=24, box_height=36, stride_x= 2, stride_y = 2):
    boxes = []
    postions=[]
    imgheight, imgwidth  = image.shape[:2]
    x_slids = (imgwidth - box_width)//stride_x
    y_slids = (imgheight-box_height)//stride_y
    for x0 in range(0, x_slids, stride_x):
        for y0 in  range(0, y_slids, stride_y):
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







