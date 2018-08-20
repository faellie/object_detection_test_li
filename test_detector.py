import tensorflow as tf
from PIL import Image
import numpy as np
class TrafficLightClassifier(object):
    def __init__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile('inference_graph/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.


        with self.detection_graph.as_default():
        # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num


def main(_):
    classifier = TrafficLightClassifier()
    image = Image.open('/opt/TF/test/210969/workcomp_210969_13_1510873007_30_49.jpg')
    (im_width, im_height) = image.size
    npImage = np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    (boxes, score, classes, num) = classifier.get_classification(npImage)
    print('boxes = ', boxes)
    print('score = ', score)
    print('class =', classes )
    print ('num = ',  num)


if __name__ == '__main__':
    tf.app.run()
