import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 读取图片
# 读取原始图像数据
image_raw_data = tf.gfile.FastGFile('../datasets/cat.jpg', 'rb').read()

with tf.Session() as sess:
    image_data = tf.image.decode_jpeg(image_raw_data)
    # 输出解码后的三维矩阵
    print(image_data.eval())
    image_data.set_shape([1797, 2673, 3])
    print(image_data.get_shape())

# 打印图片
with tf.Session() as sess:
    plt.imshow(image_data.eval())
    plt.show()


# 重新调整图片大小
# tf.image.resize_images函数封装了四种方法
# method = 0, 双线性插值        =1 最近邻插值法
#        = 2， 双三次插值法      =3 面积插值法

with tf.Session() as sess:
    # 推荐在调整图片大小前，先将图片转为0-1范围的实数
    # 如果是整数类型，API也是先转换成实数在调整，这样多次处理，会导致精度损失，0-1有利于后续处理
    image_float = tf.image.convert_image_dtype(image_data, tf.float32)
    resized = tf.image.resize_images(image_float, [300, 300], method=0)
    plt.imshow(resized.eval())
    plt.show()


# 裁剪和填充图片