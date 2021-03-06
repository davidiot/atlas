import numpy as np
import tensorflow as tf
from bounding_boxer import BOX_LABELS


class NeuralNetwork(object):
    def conv2d(self, input, filter_shape, scope_name, strides=[1, 1, 1, 1]):
        xavier_initializer = tf.contrib.layers.xavier_initializer
        with tf.variable_scope(scope_name):
            W = tf.get_variable(initializer=xavier_initializer(uniform=False),
                                name="W",
                                shape=filter_shape)
            b = tf.get_variable(initializer=xavier_initializer(uniform=False),
                                name="b",
                                shape=[filter_shape[3]])
            out = tf.nn.conv2d(input, W, padding="SAME", strides=strides)
            out = tf.nn.bias_add(out, b)
            return out

    def conv2d_relu(self, input, filter_shape, scope_name, strides=[1, 1, 1, 1]):
        xavier_initializer = tf.contrib.layers.xavier_initializer
        with tf.variable_scope(scope_name):
            W = tf.get_variable(initializer=xavier_initializer(uniform=False),
                                name="W",
                                shape=filter_shape)
            b = tf.get_variable(initializer=xavier_initializer(uniform=False),
                                name="b",
                                shape=[filter_shape[3]])
            out = tf.nn.conv2d(input, W, padding="SAME", strides=strides)
            out = tf.nn.bias_add(out, b)
            out = tf.nn.relu(out, name="out")
            return out

    def maxpool2d(self, input, scope_name, pool_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
        with tf.variable_scope(scope_name):
            out = tf.nn.max_pool(input,
                                 ksize=pool_shape,
                                 name="out",
                                 padding="SAME",
                                 strides=strides)
            return out

    def dropout(self, input, keep_prob, scope_name):
        with tf.variable_scope(scope_name):
            out = tf.nn.dropout(input, keep_prob, name="out")
            return out

    def fc(self, input, output_shape, scope_name):
        xavier_initializer = tf.contrib.layers.xavier_initializer
        with tf.variable_scope(scope_name):
            input_shape = input.shape[1:]
            input_shape = int(np.prod(input_shape))
            W = tf.get_variable(name="W",
                                shape=[input_shape, output_shape],
                                initializer=xavier_initializer(uniform=False))
            b = tf.get_variable(name="b",
                                shape=[output_shape],
                                initializer=xavier_initializer(uniform=False))
            input = tf.reshape(input, [-1, input_shape])
            # out = tf.nn.relu(tf.add(tf.matmul(input, W), b), name="out")
            out = tf.add(tf.matmul(input, W), b, name="out")
            return out

    def deconv2d(self, input, filter_shape, num_outputs, scope_name, strides=[1, 1]):
        xavier_initializer = tf.contrib.layers.xavier_initializer
        xavier_initializer_conv2d = tf.contrib.layers.xavier_initializer_conv2d
        with tf.variable_scope(scope_name):
            out = tf.contrib.layers.conv2d_transpose(input,
                                                     # activation_fn=tf.nn.relu,
                                                     activation_fn=None,
                                                     biases_initializer=xavier_initializer(uniform=False),
                                                     kernel_size=filter_shape,
                                                     num_outputs=num_outputs,
                                                     padding="SAME",
                                                     stride=strides,
                                                     weights_initializer=xavier_initializer_conv2d(uniform=False))
            out = tf.identity(out, name="out")
            return out

    def upsample(self, input, scope_name, factor=[2, 2]):
        size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
        with tf.variable_scope(scope_name):
            out = tf.image.resize_bilinear(input, size=size, align_corners=None, name="out")
            return out


class ConvEncoder(NeuralNetwork):
    def __init__(self, input_shape, keep_prob, scope_name="encoder"):
        self.input_shape = input_shape
        self.keep_prob = keep_prob
        self.scope_name = scope_name

    def build_graph(self, input):
        with tf.variable_scope(self.scope_name):
            conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, 8], scope_name="conv1")  # (232, 196, 8)
            pool1 = self.maxpool2d(conv1, scope_name="pool1")  # (116, 98, 8)
            drop1 = self.dropout(pool1, keep_prob=self.keep_prob, scope_name="drop1")
            conv2 = self.conv2d_relu(drop1, filter_shape=[5, 5, 8, 16], scope_name="conv2")  # (116, 98, 16)
            pool2 = self.maxpool2d(conv2, scope_name="pool2")  # (58, 49, 16)
            drop2 = self.dropout(pool2, keep_prob=self.keep_prob, scope_name="drop2")
            drop2 = tf.reshape(drop2, shape=[-1, 58 * 49 * 16])  # (45472,)
            fc1 = self.fc(drop2, output_shape=1024, scope_name="fc1")
            drop3 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop3")
            fc2 = self.fc(drop3, output_shape=256, scope_name="fc2")
            out = tf.identity(fc2, name="out")

        return out


class BoundingConvEncoder(NeuralNetwork):
    def __init__(self, input_shape, keep_prob, scope_name="encoder"):
        self.input_shape = input_shape
        self.keep_prob = keep_prob
        self.scope_name = scope_name

    def build_graph(self, input):
        with tf.variable_scope(self.scope_name):
            conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, 8], scope_name="conv1")  # (232, 196, 8)
            pool1 = self.maxpool2d(conv1, scope_name="pool1")  # (116, 98, 8)
            drop1 = self.dropout(pool1, keep_prob=self.keep_prob, scope_name="drop1")
            conv2 = self.conv2d_relu(drop1, filter_shape=[5, 5, 8, 16], scope_name="conv2")  # (116, 98, 16)
            pool2 = self.maxpool2d(conv2, scope_name="pool2")  # (58, 49, 16)
            drop2 = self.dropout(pool2, keep_prob=self.keep_prob, scope_name="drop2")
            drop2 = tf.reshape(drop2, shape=[-1, 58 * 49 * 16])  # (45472,)
            fc1 = self.fc(drop2, output_shape=1024, scope_name="fc1")
            drop3 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop3")
            fc2 = self.fc(drop3, output_shape=256, scope_name="fc2")
            drop4 = self.dropout(fc2, keep_prob=self.keep_prob, scope_name="drop4")
            fc3 = self.fc(drop4, output_shape=3*len(BOX_LABELS), scope_name="fc3")
            out = tf.identity(fc3, name="out")
        return out


class DeconvDecoder(NeuralNetwork):
    def __init__(self, keep_prob, output_shape, scope_name="decoder"):
        self.keep_prob = keep_prob
        self.output_shape = output_shape
        self.scope_name = scope_name

    def build_graph(self, input):
        with tf.variable_scope(self.scope_name):
            fc1 = self.fc(input, output_shape=1024, scope_name="fc1")
            drop1 = self.dropout(fc1, keep_prob=self.keep_prob, scope_name="drop1")
            fc2 = self.fc(drop1, output_shape=58 * 49 * 16, scope_name="fc2")
            drop2 = self.dropout(fc2, keep_prob=self.keep_prob, scope_name="drop2")
            drop2 = tf.reshape(drop2, shape=[-1, 58, 49, 16])
            up1 = self.upsample(drop2, scope_name="up1", factor=[2, 2])  # (116, 98, 16)
            deconv1 = self.deconv2d(up1, filter_shape=[5, 5], num_outputs=8, scope_name="deconv1")  # (116, 98, 8)
            up2 = self.upsample(deconv1, scope_name="up2", factor=[2, 2])
            deconv2 = self.deconv2d(up2, filter_shape=[3, 3], num_outputs=1, scope_name="deconv2")  # (232, 196, 1)
            out = tf.identity(deconv2, name="out")

        return out


class UNetS(NeuralNetwork):
    def __init__(self, input_shape, keep_prob, output_shape, base_size, scope_name="unet_s"):
        self.input_shape = input_shape
        self.keep_prob = keep_prob
        self.output_shape = output_shape
        self.scope_name = scope_name
        self.base_size = base_size

    def build_graph(self, input):
        base = 32 if self.base_size is None else self.base_size
        with tf.variable_scope(self.scope_name):
            # Let input_shape = (x, y)
            # Convolution
            conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, base], scope_name="conv1")  # (b, x, y, base)
            drop1 = self.dropout(conv1, keep_prob=self.keep_prob, scope_name="drop1")
            conv2 = self.conv2d_relu(drop1, filter_shape=[3, 3, base, base], scope_name="conv2")  # (b, x, y, base)
            drop2 = self.dropout(conv2, keep_prob=self.keep_prob, scope_name="drop2")

            pool1 = self.maxpool2d(drop2, scope_name="pool1")  # (b, x/1, y/2, base)
            conv3 = self.conv2d_relu(
                pool1, filter_shape=[3, 3, base, base*2], scope_name="conv3")  # (b, x/2, y/2, base*2)
            drop3 = self.dropout(conv3, keep_prob=self.keep_prob, scope_name="drop3")
            conv4 = self.conv2d_relu(
                drop3, filter_shape=[3, 3, base*2, base*2], scope_name="conv4")  # (b, x/2, y/2, base*2)
            drop4 = self.dropout(conv4, keep_prob=self.keep_prob, scope_name="drop4")

            # Deconvolution

            up2 = self.upsample(drop4, scope_name="up2", factor=[2, 2])  # (b, x, y, base*2)
            deconv2 = self.deconv2d(
                up2, filter_shape=[2, 2], num_outputs=base, scope_name="deconv2")  # (b, x, y, base)
            concat2 = tf.concat([drop2, deconv2], axis=3)  # (b, x, y, base*2)
            conv9 = self.conv2d_relu(
                concat2, filter_shape=[3, 3, base*2, base], scope_name="conv9")  # (b, x, y, base)
            drop9 = self.dropout(conv9, keep_prob=self.keep_prob, scope_name="drop9")
            conv10 = self.conv2d_relu(
                drop9, filter_shape=[3, 3, base, base], scope_name="conv10")  # (b, x, y, base)
            drop10 = self.dropout(conv10, keep_prob=self.keep_prob, scope_name="drop10")

            conv11 = self.conv2d(drop10, filter_shape=[1, 1, base, 1], scope_name="conv11")  # (b, x, y, 1)
            out = tf.identity(conv11, name="out")

        return out


class UNetM(NeuralNetwork):
    def __init__(self, input_shape, keep_prob, output_shape, base_size, scope_name="unet_m"):
        self.input_shape = input_shape
        self.keep_prob = keep_prob
        self.output_shape = output_shape
        self.scope_name = scope_name
        self.base_size = base_size

    def build_graph(self, input):
        base = 16 if self.base_size is None else self.base_size
        with tf.variable_scope(self.scope_name):
            # Let input_shape = (x, y)
            # Convolution
            conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, base], scope_name="conv1")  # (b, x, y, base)
            drop1 = self.dropout(conv1, keep_prob=self.keep_prob, scope_name="drop1")
            conv2 = self.conv2d_relu(drop1, filter_shape=[3, 3, base, base], scope_name="conv2")  # (b, x, y, base)
            drop2 = self.dropout(conv2, keep_prob=self.keep_prob, scope_name="drop2")

            pool1 = self.maxpool2d(drop2, scope_name="pool1")  # (b, x/1, y/2, base)
            conv3 = self.conv2d_relu(
                pool1, filter_shape=[3, 3, base, base*2], scope_name="conv3")  # (b, x/2, y/2, base*2)
            drop3 = self.dropout(conv3, keep_prob=self.keep_prob, scope_name="drop3")
            conv4 = self.conv2d_relu(
                drop3, filter_shape=[3, 3, base*2, base*2], scope_name="conv4")  # (b, x/2, y/2, base*2)
            drop4 = self.dropout(conv4, keep_prob=self.keep_prob, scope_name="drop4")

            pool2 = self.maxpool2d(conv4, scope_name="pool2")  # (b, x/4, y/4, base*2)
            conv5 = self.conv2d_relu(
                pool2, filter_shape=[3, 3, base*2, base*4], scope_name="conv5")  # (b, x/4, y/4, base*4)
            drop5 = self.dropout(conv5, keep_prob=self.keep_prob, scope_name="drop5")
            conv6 = self.conv2d_relu(
                drop5, filter_shape=[3, 3, base*4, base*4], scope_name="conv6")  # (b, x/4, y/4, base*4)
            drop6 = self.dropout(conv6, keep_prob=self.keep_prob, scope_name="drop6")

            # Deconvolution
            up1 = self.upsample(drop6, scope_name="up1", factor=[2, 2])  # (b, x/2, y/2, base*4)
            deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=base*2,
                                    scope_name="deconv1")  # (b, x/2, y/2, base*2)
            concat1 = tf.concat([drop4, deconv1], axis=3)  # (b, x/2, y/2, base*4)
            conv7 = self.conv2d_relu(
                concat1, filter_shape=[3, 3, base*4, base*2], scope_name="conv7")  # (b, x/2, y/2, base*2)
            drop7 = self.dropout(conv7, keep_prob=self.keep_prob, scope_name="drop7")
            conv8 = self.conv2d_relu(
                drop7, filter_shape=[3, 3, base*2, base*2], scope_name="conv8")  # (b, x/2, y/2, base*2)
            drop8 = self.dropout(conv8, keep_prob=self.keep_prob, scope_name="drop8")

            up2 = self.upsample(drop8, scope_name="up2", factor=[2, 2])  # (b, x, y, base*2)
            deconv2 = self.deconv2d(
                up2, filter_shape=[2, 2], num_outputs=base, scope_name="deconv2")  # (b, x, y, base)
            concat2 = tf.concat([drop2, deconv2], axis=3)  # (b, x, y, base*2)
            conv9 = self.conv2d_relu(
                concat2, filter_shape=[3, 3, base*2, base], scope_name="conv9")  # (b, x, y, base)
            drop9 = self.dropout(conv9, keep_prob=self.keep_prob, scope_name="drop9")
            conv10 = self.conv2d_relu(
                drop9, filter_shape=[3, 3, base, base], scope_name="conv10")  # (b, x, y, base)
            drop10 = self.dropout(conv10, keep_prob=self.keep_prob, scope_name="drop10")

            conv11 = self.conv2d(drop10, filter_shape=[1, 1, base, 1], scope_name="conv11")  # (b, x, y, 1)
            out = tf.identity(conv11, name="out")

        return out


class UNetL(NeuralNetwork):
    def __init__(self, input_shape, keep_prob, output_shape, base_size, scope_name="unet_l"):
        self.input_shape = input_shape
        self.keep_prob = keep_prob
        self.output_shape = output_shape
        self.scope_name = scope_name
        self.base_size = base_size

    def build_graph(self, input):
        base = 8 if self.base_size is None else self.base_size
        with tf.variable_scope(self.scope_name):
            # Let input_shape = (x, y)
            # Convolution
            conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, base], scope_name="conv1")  # (b, x, y, base)
            drop1 = self.dropout(conv1, keep_prob=self.keep_prob, scope_name="drop1")
            conv2 = self.conv2d_relu(drop1, filter_shape=[3, 3, base, base], scope_name="conv2")  # (b, x, y, base)
            drop2 = self.dropout(conv2, keep_prob=self.keep_prob, scope_name="drop2")

            pool1 = self.maxpool2d(drop2, scope_name="pool1")  # (b, x/1, y/2, base)
            conv3 = self.conv2d_relu(
                pool1, filter_shape=[3, 3, base, base*2], scope_name="conv3")  # (b, x/2, y/2, base*2)
            drop3 = self.dropout(conv3, keep_prob=self.keep_prob, scope_name="drop3")
            conv4 = self.conv2d_relu(
                drop3, filter_shape=[3, 3, base*2, base*2], scope_name="conv4")  # (b, x/2, y/2, base*2)
            drop4 = self.dropout(conv4, keep_prob=self.keep_prob, scope_name="drop4")

            pool2 = self.maxpool2d(conv4, scope_name="pool2")  # (b, x/4, y/4, base*2)
            conv5 = self.conv2d_relu(
                pool2, filter_shape=[3, 3, base*2, base*4], scope_name="conv5")  # (b, x/4, y/4, base*4)
            drop5 = self.dropout(conv5, keep_prob=self.keep_prob, scope_name="drop5")
            conv6 = self.conv2d_relu(
                drop5, filter_shape=[3, 3, base*4, base*4], scope_name="conv6")  # (b, x/4, y/4, base*4)
            drop6 = self.dropout(conv6, keep_prob=self.keep_prob, scope_name="drop6")

            poolx1 = self.maxpool2d(conv6, scope_name="poolx1")  # (b, x/8, y/8, base*4)
            convx1 = self.conv2d_relu(
                poolx1, filter_shape=[3, 3, base*4, base*8], scope_name="convx1")  # (b, x/8, y/8, base*8)
            dropx1 = self.dropout(convx1, keep_prob=self.keep_prob, scope_name="dropx1")
            convx2 = self.conv2d_relu(
                dropx1, filter_shape=[3, 3, base*8, base*8], scope_name="convx2")  # (b, x/8, y/8, base*8)
            dropx2 = self.dropout(convx2, keep_prob=self.keep_prob, scope_name="dropx2")

            # Deconvolution
            upx1 = self.upsample(dropx2, scope_name="upx1", factor=[2, 2])  # (b, x/4, y/4, base*8)
            deconvx1 = self.deconv2d(upx1, filter_shape=[2, 2], num_outputs=base*4,
                                     scope_name="deconvx1")  # (b, x/4, y/4, base*4)
            concatx1 = tf.concat([drop6, deconvx1], axis=3)  # (b, x/4, y/4, base*8)
            convx3 = self.conv2d_relu(
                concatx1, filter_shape=[3, 3, base*8, base*4], scope_name="convx3")  # (b, x/2, y/2, base*2)
            dropx3 = self.dropout(convx3, keep_prob=self.keep_prob, scope_name="dropx3")
            convx4 = self.conv2d_relu(
                dropx3, filter_shape=[3, 3, base*4, base*4], scope_name="convx4")  # (b, x/2, y/2, base*2)
            dropx4 = self.dropout(convx4, keep_prob=self.keep_prob, scope_name="dropx4")

            up1 = self.upsample(dropx4, scope_name="up1", factor=[2, 2])  # (b, x/2, y/2, base*4)
            deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=base*2,
                                    scope_name="deconv1")  # (b, x/2, y/2, base*2)
            concat1 = tf.concat([drop4, deconv1], axis=3)  # (b, x/2, y/2, base*4)
            conv7 = self.conv2d_relu(
                concat1, filter_shape=[3, 3, base*4, base*2], scope_name="conv7")  # (b, x/2, y/2, base*2)
            drop7 = self.dropout(conv7, keep_prob=self.keep_prob, scope_name="drop7")
            conv8 = self.conv2d_relu(
                drop7, filter_shape=[3, 3, base*2, base*2], scope_name="conv8")  # (b, x/2, y/2, base*2)
            drop8 = self.dropout(conv8, keep_prob=self.keep_prob, scope_name="drop8")

            up2 = self.upsample(drop8, scope_name="up2", factor=[2, 2])  # (b, x, y, base*2)
            deconv2 = self.deconv2d(
                up2, filter_shape=[2, 2], num_outputs=base, scope_name="deconv2")  # (b, x, y, base)
            concat2 = tf.concat([drop2, deconv2], axis=3)  # (b, x, y, base*2)
            conv9 = self.conv2d_relu(
                concat2, filter_shape=[3, 3, base*2, base], scope_name="conv9")  # (b, x, y, base)
            drop9 = self.dropout(conv9, keep_prob=self.keep_prob, scope_name="drop9")
            conv10 = self.conv2d_relu(
                drop9, filter_shape=[3, 3, base, base], scope_name="conv10")  # (b, x, y, base)
            drop10 = self.dropout(conv10, keep_prob=self.keep_prob, scope_name="drop10")

            conv11 = self.conv2d(drop10, filter_shape=[1, 1, base, 1], scope_name="conv11")  # (b, x, y, 1)
            out = tf.identity(conv11, name="out")

        return out


class UNetXL(NeuralNetwork):
    def __init__(self, input_shape, keep_prob, output_shape, base_size, scope_name="unet_xl"):
        self.input_shape = input_shape
        self.keep_prob = keep_prob
        self.output_shape = output_shape
        self.scope_name = scope_name
        self.base_size = base_size

    def build_graph(self, input):
        base = 4 if self.base_size is None else self.base_size
        with tf.variable_scope(self.scope_name):
            # Let input_shape = (x, y)
            # Convolution
            conv1 = self.conv2d_relu(input, filter_shape=[3, 3, 1, base], scope_name="conv1")  # (b, x, y, base)
            drop1 = self.dropout(conv1, keep_prob=self.keep_prob, scope_name="drop1")
            conv2 = self.conv2d_relu(drop1, filter_shape=[3, 3, base, base], scope_name="conv2")  # (b, x, y, base)
            drop2 = self.dropout(conv2, keep_prob=self.keep_prob, scope_name="drop2")

            pool1 = self.maxpool2d(drop2, scope_name="pool1")  # (b, x/1, y/2, base)
            conv3 = self.conv2d_relu(
                pool1, filter_shape=[3, 3, base, base*2], scope_name="conv3")  # (b, x/2, y/2, base*2)
            drop3 = self.dropout(conv3, keep_prob=self.keep_prob, scope_name="drop3")
            conv4 = self.conv2d_relu(
                drop3, filter_shape=[3, 3, base*2, base*2], scope_name="conv4")  # (b, x/2, y/2, base*2)
            drop4 = self.dropout(conv4, keep_prob=self.keep_prob, scope_name="drop4")

            pool2 = self.maxpool2d(conv4, scope_name="pool2")  # (b, x/4, y/4, base*2)
            conv5 = self.conv2d_relu(
                pool2, filter_shape=[3, 3, base*2, base*4], scope_name="conv5")  # (b, x/4, y/4, base*4)
            drop5 = self.dropout(conv5, keep_prob=self.keep_prob, scope_name="drop5")
            conv6 = self.conv2d_relu(
                drop5, filter_shape=[3, 3, base*4, base*4], scope_name="conv6")  # (b, x/4, y/4, base*4)
            drop6 = self.dropout(conv6, keep_prob=self.keep_prob, scope_name="drop6")

            poolx1 = self.maxpool2d(conv6, scope_name="poolx1")  # (b, x/8, y/8, base*4)
            convx1 = self.conv2d_relu(
                poolx1, filter_shape=[3, 3, base*4, base*8], scope_name="convx1")  # (b, x/8, y/8, base*8)
            dropx1 = self.dropout(convx1, keep_prob=self.keep_prob, scope_name="dropx1")
            convx2 = self.conv2d_relu(
                dropx1, filter_shape=[3, 3, base*8, base*8], scope_name="convx2")  # (b, x/8, y/8, base*8)
            dropx2 = self.dropout(convx2, keep_prob=self.keep_prob, scope_name="dropx2")

            poolxx1 = self.maxpool2d(convx2, scope_name="poolxx1")  # (b, x/16, y/16, base*8)
            convxx1 = self.conv2d_relu(
                poolxx1, filter_shape=[3, 3, base*8, base*16], scope_name="convxx1")  # (b, x/16, y/16, base*16)
            dropxx1 = self.dropout(convxx1, keep_prob=self.keep_prob, scope_name="dropxx1")
            convxx2 = self.conv2d_relu(
                dropxx1, filter_shape=[3, 3, base*16, base*16], scope_name="convxx2")  # (b, x/16, y/16, base*16)
            dropxx2 = self.dropout(convxx2, keep_prob=self.keep_prob, scope_name="dropxx2")

            # Deconvolution
            upxx1 = self.upsample(dropxx2, scope_name="upxx1", factor=[2, 2])  # (b, x/8, y/8, base*16)
            deconvxx1 = self.deconv2d(upxx1, filter_shape=[2, 2], num_outputs=base*8,
                                     scope_name="deconvxx1")  # (b, x/8, y/8, base*8)
            concatxx1 = tf.concat([dropx2, deconvxx1], axis=3)  # (b, x/8, y/8, base*16)
            convxx3 = self.conv2d_relu(
                concatxx1, filter_shape=[3, 3, base*16, base*8], scope_name="convxx3")  # (b, x/4, y/4, base*4)
            dropxx3 = self.dropout(convxx3, keep_prob=self.keep_prob, scope_name="dropxx3")
            convxx4 = self.conv2d_relu(
                dropxx3, filter_shape=[3, 3, base*8, base*8], scope_name="convxx4")  # (b, x/4, y/4, base*4)
            dropxx4 = self.dropout(convxx4, keep_prob=self.keep_prob, scope_name="dropxx4")

            upx1 = self.upsample(dropxx4, scope_name="upx1", factor=[2, 2])  # (b, x/4, y/4, base*8)
            deconvx1 = self.deconv2d(upx1, filter_shape=[2, 2], num_outputs=base*4,
                                     scope_name="deconvx1")  # (b, x/4, y/4, base*4)
            concatx1 = tf.concat([drop6, deconvx1], axis=3)  # (b, x/4, y/4, base*8)
            convx3 = self.conv2d_relu(
                concatx1, filter_shape=[3, 3, base*8, base*4], scope_name="convx3")  # (b, x/2, y/2, base*2)
            dropx3 = self.dropout(convx3, keep_prob=self.keep_prob, scope_name="dropx3")
            convx4 = self.conv2d_relu(
                dropx3, filter_shape=[3, 3, base*4, base*4], scope_name="convx4")  # (b, x/2, y/2, base*2)
            dropx4 = self.dropout(convx4, keep_prob=self.keep_prob, scope_name="dropx4")

            up1 = self.upsample(dropx4, scope_name="up1", factor=[2, 2])  # (b, x/2, y/2, base*4)
            deconv1 = self.deconv2d(up1, filter_shape=[2, 2], num_outputs=base*2,
                                    scope_name="deconv1")  # (b, x/2, y/2, base*2)
            concat1 = tf.concat([drop4, deconv1], axis=3)  # (b, x/2, y/2, base*4)
            conv7 = self.conv2d_relu(
                concat1, filter_shape=[3, 3, base*4, base*2], scope_name="conv7")  # (b, x/2, y/2, base*2)
            drop7 = self.dropout(conv7, keep_prob=self.keep_prob, scope_name="drop7")
            conv8 = self.conv2d_relu(
                drop7, filter_shape=[3, 3, base*2, base*2], scope_name="conv8")  # (b, x/2, y/2, base*2)
            drop8 = self.dropout(conv8, keep_prob=self.keep_prob, scope_name="drop8")

            up2 = self.upsample(drop8, scope_name="up2", factor=[2, 2])  # (b, x, y, base*2)
            deconv2 = self.deconv2d(
                up2, filter_shape=[2, 2], num_outputs=base, scope_name="deconv2")  # (b, x, y, base)
            concat2 = tf.concat([drop2, deconv2], axis=3)  # (b, x, y, base*2)
            conv9 = self.conv2d_relu(
                concat2, filter_shape=[3, 3, base*2, base], scope_name="conv9")  # (b, x, y, base)
            drop9 = self.dropout(conv9, keep_prob=self.keep_prob, scope_name="drop9")
            conv10 = self.conv2d_relu(
                drop9, filter_shape=[3, 3, base, base], scope_name="conv10")  # (b, x, y, base)
            drop10 = self.dropout(conv10, keep_prob=self.keep_prob, scope_name="drop10")

            conv11 = self.conv2d(drop10, filter_shape=[1, 1, base, 1], scope_name="conv11")  # (b, x, y, 1)
            out = tf.identity(conv11, name="out")

        return out
