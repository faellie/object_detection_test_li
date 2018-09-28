import tensorflow as tf
import os

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1] #all to the last column
label_name = column_names[-1]  # the last column

train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    32,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
iter = train_dataset.make_one_shot_iterator()
features, labels = iter.get_next()
print('features = ', features, features.numpy())


