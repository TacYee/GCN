import numpy as np
import tensorflow as tf
from lib.graph import Graph
tf.random.set_seed(5)

MyGraph = Graph("uniform", 19, 2)

class GraphConv(tf.keras.layers.Layer):
    """Basic graphic convolutional layer for keras"""
    def __init__(self, filters, t_kernels=1, t_strides=1, **kwargs):
        super(GraphConv, self).__init__()
        self.filters = filters
        self.t_kernels = t_kernels // 2 * 2 + 1
        self.A = MyGraph.A
        kwargs["padding"] = "same"
        kwargs["strides"] = (t_strides, 1)
        self.kwargs = kwargs
        self.Conv2D = None
        self.Reshape = None

    def build(self, input_shape):
        A_shape = self.A.shape
        try:
            x_shape = input_shape
        except ValueError:
            raise ValueError("Expected input_shape are x.shape!")

        if len(x_shape) != 4:
            raise ValueError("Input x shape should be 3 dimension in "
                             "[timesteps, nodes, channels], but find "
                             "{} of {}".format(len(x_shape), x_shape[1:]))
        if len(A_shape) != 3:
            raise ValueError("Input A shape should be 3 dimension in "
                             "[label_number, nodes, nodes], but find "
                             "{} of {}".format(len(A_shape), A_shape))
        self.Conv2D = tf.keras.layers.Conv2D(self.filters * A_shape[0],
                                             (self.t_kernels, 1),
                                             **self.kwargs)
        super(GraphConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        try:
            x = inputs
        except ValueError:
            raise ValueError("Expected inputs are x!")
        x = self.Conv2D(x)
        self.kernel_size = self.A.shape[0]
        t = x.shape[1]
        v = x.shape[2]
        kc = x.shape[3]

        x = tf.reshape(x, (-1, t, v, self.kernel_size, kc//self.kernel_size))

        # compute the graphic conv:
        # f_out = sum_j{A_j * (X * W_j)}
        x = tf.einsum("ntvkc,kvw->ntwc", x, self.A)
        return x

    def compute_output_shape(self, input_shape):
        x_shape = input_shape
        return [np.asarray(
            self.Conv2D.compute_output_shape(x_shape).tolist()[:-1] +
            [self.filters])]


