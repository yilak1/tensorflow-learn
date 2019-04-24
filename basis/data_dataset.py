import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
import os

def get_filenames(is_training, data_dir):
  """
  Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if is_training:
    return [os.path.join(data_dir, 'voc_train.record')]
  else:
    return [os.path.join(data_dir, 'voc_val.record')]

def parse_record(raw_record):
    """Parse PASCAL image and label from a tf record."""
    keys_to_features = {
      'image/height':
      tf.FixedLenFeature((), tf.int64),
      'image/width':
      tf.FixedLenFeature((), tf.int64),
      'image/encoded':
      # encoded:编码
      tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'label/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'label/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    # height = tf.cast(parsed['image/height'], tf.int32)
    # width = tf.cast(parsed['image/width'], tf.int32)

    image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(
      tf.reshape(parsed['label/encoded'], shape=[]), 1)
    label = tf.cast((tf.image.convert_image_dtype(label, dtype=tf.uint8)),tf.int32)
    label.set_shape([None, None, 1])

    return image, label

is_training=True
data_dir="/media/disk/lds/LDS/tf-deeplab-v3/dataset"
print(get_filenames(is_training, data_dir))
dataset = tf.data.Dataset.from_tensor_slices(['/media/disk/lds/LDS/tf-deeplab-v3/dataset/voc_train.record'])
print(dataset)
# print(os.path.join('/media/disk/lds/LDS/tf-deeplab-v3/dataset','voc_val.record'))
# print(os.path.exists('/media/disk/lds/LDS/tf-deeplab-v3/dataset/voc_val.record'))
# dataset = tf.data.TFRecordDataset("/media/disk/lds/LDS/tf-deeplab-v3/dataset/voc_val.record")
#
# print(dataset)
# dataset = dataset.flat_map(tf.data.TFRecordDataset)

