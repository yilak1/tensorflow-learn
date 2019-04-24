import tempfile
import tensorflow as tf
import os

# 定义运行的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# 1.从数组创建数据集
input_data = [1, 2, 3, 5, 8]
# from_tensor_slices(tensor)参数可以为tensor,list,and so on
dataset = tf.data.Dataset.from_tensor_slices(input_data)
# 定义iterator迭代器
iterator = dataset.make_one_shot_iterator()
# 取出元素
x = iterator.get_next()
y = x * x

with tf.Session() as sess:
    # 每次取出一个元素
    for i in range(len(input_data)):
        print(sess.run(y))

# 读取文本文件里的数据
with open("./test1.txt", "w") as file:
    file.write("File1, line1.\n")
    file.write("File1, line2.\n")

with open("./test2.txt", "w") as file:
    file.write("File2, line1.\n")
    file.write("File2, line2.\n")

# 2.从文本文件创建数据集。这里可以提供多个文件。
input_files = ["./test1.txt", "./test2.txt"]
dataset = tf.data.TextLineDataset(input_files)

# 定义迭代器
iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()
with tf.Session() as sess:
    for i in range(4):
        print(sess.run(x))


# 3.解析TFRecord文件里的数据。读取文件为本章第一节创建的文件
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw':tf.FixedLenFeature([], tf.string),
            'pixels':tf.FixedLenFeature([], tf.int64),
            'label':tf.FixedLenFeature([], tf.int64)
        })
    print(features)
    print(features['image_raw'])
    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images = tf.reshape(retyped_images, [784])
    labels = tf.cast(features['label'], tf.int32)
    return images, labels


input_files = ["output.tfrecords"]
dataset = tf.data.TFRecordDataset(input_files)
# map(map_func, num_parallel_calls=None)
dataset = dataset.map(parser)
iterator = dataset.make_one_shot_iterator()
image, label = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        x, y = sess.run([image, label])
        print(y)


# 4.使用initializable_iterator 来动态创建初始化数据集
input_files = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parser)
# 使用初始化迭代器make_initializable_iterator()，可以动态指定参数
iterator = dataset.make_initializable_iterator()
image, label = iterator.get_next()

with tf.Session() as sess:
    # 首先初始化iterator 并给出input_files的值
    sess.run(iterator.initializer, feed_dict={input_files:["output.tfrecords"]})
    # 遍历所有数据一个epoch。当遍历结束时，程序会抛出OutOfRangeError
    while True:
        try:
            x, y = sess.run([image, label])
        except tf.errors.OutOfRangeError:
            break
