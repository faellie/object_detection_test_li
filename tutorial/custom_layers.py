import tensorflow as tf
tfe = tf.contrib.eager

tf.enable_eager_execution()

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[input_shape[-1].value,
                                               self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
buildinput =tf.random_normal([10, 5])
buildinput[-1]
print('buildinput:', buildinput)
#print('buildinput[-2]:', buildinput[-2,])
#print('buildinput[-1].value:', buildinput[-1].value)
print('layer : ' , layer(tf.zeros([10, 5])))
print('layer.variables', layer.variables)
