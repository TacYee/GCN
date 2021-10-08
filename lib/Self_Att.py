import numpy as np
import tensorflow as tf
tf.random.set_seed(5)

class Self_Attn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Self_Attn, self).__init__()
        self.Conv2D = None
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.softmax = None
        self.gamma = 0

    def build(self, input_shape):
        self.query_conv = tf.keras.layers.Conv2D(1, (1, 1))
        self.key_conv = tf.keras.layers.Conv2D(1, (1, 1))
        self.value_conv = tf.keras.layers.Conv2D(1, (1, 1))
        self.gamma = tf.Variable(tf.zeros(1))

        super(Self_Attn, self).build(input_shape)

    def call(self, inputs, **kwargs):
            x = inputs      #input feature maps(B * H * W),H->T,W->N
            height = x.shape[1]
            width = x.shape[2]
            x = tf.reshape(x, (-1, height, width, 1))
            query = self.query_conv(x)
            query = tf.reshape(query, (-1, height, width))
            key = self.key_conv(x)
            key = tf.reshape(key, (-1, height, width))
            key = tf.transpose(key, [0, 2, 1])
            energy = tf.matmul(query, key)
            attention = tf.keras.activations.softmax(energy, axis=-1)
            value = self.value_conv(x)
            value = tf.reshape(value, (-1, height, width))
            value = tf.transpose(value, [0, 2, 1])
            out = tf.matmul(value, tf.transpose(attention, [0, 2, 1]))
            out = tf.transpose(out, [0, 2, 1])   #output feature maps(B * H * W),H->T,W->N
            out = self.gamma * out + inputs
            out_pool = tf.reduce_mean(out, axis=-1)

            return out

class Self_Attn_3D(tf.keras.layers.Layer):
    def __init__(self, in_channel, filters, **kwargs):
        super(Self_Attn_3D, self).__init__()
        self.in_channel = in_channel
        self.filters = filters
        self.Conv2D = None
        self.query_conv = None
        self.key_conv = None
        self.value_conv = None
        self.softmax = None
        self.gamma = 0

    def build(self, input_shape):
        self.query_conv = tf.keras.layers.Conv2D(self.filters, (1, 1))
        self.key_conv = tf.keras.layers.Conv2D(self.filters, (1, 1))
        self.value_conv = tf.keras.layers.Conv2D(self.in_channel, (1, 1))
        self.gamma = tf.Variable(tf.zeros(1))

        super(Self_Attn_3D, self).build(input_shape)

    def call(self, inputs, **kwargs):
            x = inputs      #input feature maps(B * H * W * C),H->T,W->N
            height = x.shape[1]
            width = x.shape[2]
            channels = self.filters
            channel_in = self.in_channel
            query = self.query_conv(x)
            query = tf.reshape(query, (-1, height*width, channels))
            key = self.key_conv(x)
            key = tf.reshape(key, (-1, height*width, channels))
            key = tf.transpose(key, [0, 2, 1])
            energy = tf.matmul(query, key)
            attention = tf.keras.activations.softmax(energy, axis=-1)
            value = self.value_conv(x)
            value = tf.reshape(value, (-1, height*width, channel_in))
            value = tf.transpose(value, [0, 2, 1])
            out = tf.matmul(value, tf.transpose(attention, [0, 2, 1]))
            out = tf.transpose(out, [0, 2, 1])
            out = tf.reshape(out, (-1, height, width, channel_in))  #output feature maps(B * H * W * C),H->T,W->N
            out = self.gamma * out + inputs
            out_pool = tf.reduce_mean(out, axis=-1)

            return out_pool



