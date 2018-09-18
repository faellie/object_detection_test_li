#from https://www.tensorflow.org/tutorials/eager/eager_basics
import tensorflow as tf

tf.enable_eager_execution()
tfe = tf.contrib.eager

print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.constant([[[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))
print(tf.random_uniform([4, 4]))
#ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
#ds_tensors = tf.data.Dataset.from_tensor_slices([[1,2], [2,2]])
ds_tensors = tf.data.Dataset.from_tensor_slices([[[1,2]], [[2,2]]])
print(ds_tensors)
for x in ds_tensors:
    print(x)

import math
from math import pi

def f(x):
    return tf.square(tf.sin(x))

def f1(x):
    return math.sin(x) * math.sin(x)



#is is a tensor
print(f(pi/2))

#.numpy set it to the number
print(f(pi/2).numpy())

#this is also just a number because we did not use tf
print(f1(pi/2))

assert f(pi/2).numpy() == 1.0


grad_f = tfe.gradients_function(f)

ds_list = tf.data.Dataset.from_tensor_slices([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
for x in ds_list :
    print(tf.abs(grad_f(pi*x)[0]).numpy())

x = tf.lin_space(-2*pi, 2*pi, 100)  # 100 points between -2PI and + 2PI

import matplotlib.pyplot as plt

plt.plot(x, f(x), label="f")
plt.plot(x, grad_f(x)[0], label="first derivative")
# plt.plot(x, grad_f(grad_f(f))(x), label="second derivative")
# plt.plot(x, grad_f(grad_f(grad_f(f)))(x), label="third derivative")
# plt.legend()
plt.show()

x = tf.zeros([10, 10])
x += 2  # This is equivalent to x = x + 2, which does not mutate the original
# value of x
print(x)
