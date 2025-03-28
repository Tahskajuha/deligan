import numpy as np
import tensorflow as tf

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.compat.v1.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.compat.v1.variable_scope(self.name) as scope:
                self.beta = tf.compat.v1.get_variable("beta", [shape[-1]],
                                    initializer=tf.compat.v1.constant_initializer(0.))
                self.gamma = tf.compat.v1.get_variable("gamma", [shape[-1]],
                                    initializer=tf.compat.v1.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

# standard convolution layer
def conv2d(x, filter_size, stride, inputFeatures, outputFeatures, name):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable("w",[filter_size,filter_size,inputFeatures, outputFeatures], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        b = tf.compat.v1.get_variable("b",[outputFeatures], initializer=tf.compat.v1.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, filters=w, strides=[1,stride,stride,1], padding="SAME") + b
        return conv

def conv_transpose(x, filter_size, stride, outputShape, name):
    with tf.compat.v1.variable_scope(name):
        # h, w, out, in
        w = tf.compat.v1.get_variable("w",[filter_size,filter_size, outputShape[-1], x.get_shape()[-1]], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
        b = tf.compat.v1.get_variable("b",[outputShape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,stride,stride,1])
        return convt

# leaky reLu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.compat.v1.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# fully-conected layer
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    with tf.compat.v1.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32, tf.compat.v1.random_normal_initializer(stddev=0.02))
        bias = tf.compat.v1.get_variable("bias", [outputFeatures], initializer=tf.compat.v1.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias
