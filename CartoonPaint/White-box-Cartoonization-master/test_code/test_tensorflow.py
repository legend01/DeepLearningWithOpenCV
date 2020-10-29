'''
Description: 测试TensorFlow开发环境
Author: HLLI8
Date: 2020-10-22 18:08:10
LastEditTime: 2020-10-29 10:10:44
LastEditors: HLLI8
'''
import tensorflow as tf #引入模块
tf.compat.v1.disable_eager_execution()
x = tf.constant([[1.0, 2.0]]) #定义一个 2 阶张量等于[[1.0,2.0]]
w = tf.constant([[3.0], [4.0]]) #定义一个 2 阶张量等于[[3.0],[4.0]]
y = tf.matmul(x, w) #实现 xw 矩阵乘法
print (y) #打印出结果
# 初始化所有变量，也就是上面定义的a/b两个变量
#tf.compat.v1.Session() as session
with tf.compat.v1.Session() as sess:
    print (sess.run(y))








