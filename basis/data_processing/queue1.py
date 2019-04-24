import tensorflow as tf
import numpy as np
import threading
import time
# 创建队列,并操作里面的元素
q = tf.FIFOQueue(2, "int32")
# 初始化队列中的元素，在使用队列之前需要明确的调用这个初始化过程
init = q.enqueue_many(([0, 10],))
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])
with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(v)


def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand()<0.1:
            print("Stoping from id: %d\n" % worker_id)
            coord.request_stop()
        else:
            print("Working on id: %d\n" % worker_id)
        time.sleep(1)


coord = tf.train.Coordinator()
threads = [threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5)]
# 启动所有线程
for t in threads:t.start()
# 等待所有线程推出
coord.join(threads)