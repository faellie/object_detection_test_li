import os
from tensorflow.keras.models import model_from_json

# Helper libraries
from tutorial.my_utils import *

#todo slide over a whole image and try to find the number
#first try to found 'five'
fullimagefile = '/opt/tmp/image/verysmall.jpg'
#fullimagefile = '/opt/tmp/image/small.jpg'
image = cv2.imread(fullimagefile)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
sliding_boxes, positions = sliding_over(image, box_width=22, box_height=32, stride_x= 4, stride_y = 4, save_to_file='')
print('number of boxes = ', len(sliding_boxes))
probs = []
detected_boxes = []
i = 0;


# load json and create model
json_file = open('/opt/tmp/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/opt/tmp/model.h5")
print("Loaded model from disk")



for box in sliding_boxes:
    i = i + 1
    x = img_to_array(box)
    image_height =36
    image_width = 24
    x = (x - 128.0) / 128.0
    x = x / 255.0
    x = x.reshape(1, image_height, image_width)
    number = loaded_model.predict_classes(x)
    probs = loaded_model.predict_proba(x)
    prob = probs[0][number][0]
    if(prob > 0.99 and number != 10):
        print('found ' + str(number))
        detected_boxes.append(positions[i-1])
        #plt.figure()
        #plt.imshow(box)
        #plt.show()
        if(0):
            cv2.imshow(str(number),  box)
            cv2.waitKey(0)
print("detected : " , len(detected_boxes))

if(1):
    # Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    # Create a Rectangle patch
    index = 0
    import random
    for (x0, y0, x1, y1) in detected_boxes:
        print((x0, y0, x1, y1) )
        rect = patches.Rectangle((x0,y0),x1-x0,y1-y0,linewidth=1,edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        index = index + 1
        #if(index > 10):
        #    break
    plt.show()
