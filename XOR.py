# -*- coding:utf-8 -*-
# !/usr/bin/python

import tensorflow as tf
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

x = tf.placeholder(tf.float32, [None, 2], name="x")
y = tf.placeholder(tf.float32, [None, 1], name='y')

w1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.constant(0.1, shape=[2]))
w2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.zeros([1]))

h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
y_hat = tf.nn.relu(tf.matmul(h1, w2) + b2)

loss = tf.reduce_mean(tf.square(y_hat - y))

train = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        for j in range(4):
            sess.run(train, feed_dict={x: np.expand_dims(X[j], 0), y: np.expand_dims(Y[j], 0)})
        loss_ = sess.run(loss, feed_dict={x: X, y: Y})
        print("step: %d, loss: %.3f" % (i, loss_))
    print("X: %r" % X)
    print("pred: %r" % sess.run(y_hat, feed_dict={x: X}))
