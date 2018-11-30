#reference from https://machinelearningmastery.com/save-load-keras-deep-learning-models/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
import numpy
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


from tutorial.my_utils import *

# load json and create model
json_file = open('/opt/tmp/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/opt/tmp/model.h5")
print("Loaded model from disk")


test_images = load_images_for_detect('/opt/tmp/image/croped')
#test_images = load_images_for_detect('/opt/tmp/out/bg')
class_names = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine', 'bg']



print('test images: ' , test_images.shape)



test_images = test_images / 255.0
if(0):
    for img in test_images:
        plt.figure()
        plt.imshow(img)
        plt.show()
if(1):
    plt.figure(figsize=(12,12))
    for i in range(min(100, len(test_images))):
        plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i])

        #now found the number and label it
        image_height =36
        image_width = 24
        x = img_to_array(test_images[i])
        x = x.reshape(1, image_height, image_width)
        number = loaded_model.predict_classes(x)
        probs = loaded_model.predict_proba(x)
        prob = probs[0][number][0]
        plt.xlabel(str(number)  +'-' + str(round(prob, 2)))

    plt.show()
if(1):
    count = 0
    image_height =36
    image_width = 24
    for i in range(len(test_images)):
        x = img_to_array(test_images[i])
        x = x.reshape(1, image_height, image_width)
        number = loaded_model.predict_classes(x)
        probs = loaded_model.predict_proba(x)
        prob = probs[0][number][0]
        if(prob > 0.9 and number != 10):
            print('found ' + str(number))
            count += 1
    print('total count ' , count)
