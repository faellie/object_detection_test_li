from tutorial.my_utils import *
import cv2

image = cv2.imread('/opt/tmp/image/verysmall.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


sliding_over(image, box_width=22, box_height=32, stride_x= 2, stride_y = 2, save_to_file='/opt/tmp/image/croped')
