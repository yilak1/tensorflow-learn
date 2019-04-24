import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 将输入转化成TFRecord格式并保存

# 定义函数转化变量类型
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 将数据转化为tf.train.Example格式
def _make_example(pixels, label, image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example

# 读取mnist训练数据。
mnist = input_data.read_data_sets("../datasets/MNIST_data",dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出包含训练数据的TFRecord文件。
with tf.python_io.TFRecordWriter('output.tfrecords') as writer:
    for index in range(num_examples):
        example = _make_example(pixels, labels[index], images[index])
        writer.write(example.SerializeToString())

print("TFRecord 文件已经保存")

# 读取mnist测试数据。
images_test = mnist.test.images
labels_test = mnist.test.labels
pixels_test = images_test.shape[1]
num_examples_test = mnist.test.num_examples

# 输出包含测试数据的TFRecord文件。
with tf.python_io.TFRecordWriter("output_test.tfrecords") as writer:
    for index in range(num_examples_test):
        example = _make_example(
            pixels_test, labels_test[index], images_test[index])
        writer.write(example.SerializeToString())
print("TFRecord测试文件已保存。")

# 读取文件
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["output.tfrecords"])
# 读取多个样例用read_up_to
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例,解析多个样例用parse_example
features = tf.parse_single_example(
    serialized_example,
    features={
        # FixedLenFeature返回解析结果为Tensor
        # tf.VarLenFeature解析结果为SparseTensor
        'image_raw':tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label':tf.FixedLenFeature([], tf.int64)
    }
)

# 可以把字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

with tf.Session() as sess:
    # 启动多线程处理数据
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 每次读取TFRecord文件中的一个example，当所有样例读取完后，在此样例中程序会再重头读取
    for i in range(10):
        image, label, pixel = sess.run([images, labels, pixels])
