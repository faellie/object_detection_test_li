import tensorflow as tf
import json
from object_detection.utils import dataset_util
import io
import os
import numpy as np
import PIL.Image as pil

flags = tf.app.flags
flags.DEFINE_string('output_path', '/opt/TF/test/elvdata/li-elv.tfrecord', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(entry):

    filename = entry['image_path'] # Filename of the image. Empty if image is not from file
    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = pil.open(encoded_png_io)
    image = np.asarray(image)
    width = int(image.shape[1])
    height = int(image.shape[0])



    # encoded_image_data = dataset_util.bytes_feature(encoded_png)
    image_format = 'png' # b'jpeg' or b'png'
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    for box in entry["rects"] :
        xmins.append(box["x1"]/640.0)
        xmaxs.append(box["x2"]/640.0)
        ymins.append(box["y1"]/480.0)
        ymaxs.append(box["y2"]/480.0)
        classes_text.append('FIVE')
        classes.append(5)


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    with open('/opt/TF/test/elvdata/elv.json') as f:
        data = json.load(f)
    for entry in data :
        tf_example = create_tf_example(entry)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
