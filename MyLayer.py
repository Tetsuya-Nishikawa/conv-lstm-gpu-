import tensorflow as tf
import numpy as np
import MyLibrary

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filter_shape, input_channels, filters, strides_shape, padding):
        super(ConvLayer, self).__init__()
        self.conv_filt_shape = [filter_shape[0], filter_shape[1],  input_channels, filters]
        self.filters = filters
        self.strides             = strides_shape
        self.padding = padding
    def build(self, input_shape):
        self.w = self.add_weight(shape=self.conv_filt_shape, initializer = tf.keras.initializers.he_normal(), trainable=True)
        self.b = self.add_weight(shape=[self.filters,], initializer = tf.keras.initializers.he_normal(), trainable=True)
    
    def compute_mask(self, inputs, mask=None):
        return mask
 
    def call(self, inputs, mask=None):
        filter_shape = [self.conv_filt_shape[0], self.conv_filt_shape[1]]
        #ストライドが2の時は、元の画像サイズに対しての半分に調整
        if self.strides[0] == 2:
            pad = -(-(-2+filter_shape[0])//2)
            inputs = MyLibrary.zero_padding(inputs, pad)
        #inputs   = batch_layer(inputs, 3)
        
        outputs = tf.nn.conv2d(inputs, self.w, strides=self.strides, padding=self.padding) + self.b
        return tf.nn.relu(outputs)

class ResNetLayer(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(ResNetLayer, self).__init__()
        self.filters = filters
        self.strides = strides
        
    def build(self, input_shape):
        input_dimention = input_shape[3]
        if self.strides[0]==2:
            self.conv1                = ConvLayer([1, 1], input_dimention, self.filters//4, self.strides, "VALID")
        else:
            self.conv1                = ConvLayer([1, 1], input_dimention, self.filters//4, self.strides, "SAME")          
        self.conv2                = ConvLayer([3, 3], self.filters//4,                 self.filters//4, [1, 1], "SAME")
        self.conv3                = ConvLayer([1, 1], self.filters//4,                 self.filters     , [1, 1], "SAME")
        if self.strides[0]==2:
            self.shortcut = ConvLayer([3, 3], input_dimention, self.filters, [2, 2], "VALID")
        else:
            self.shortcut = ConvLayer([3, 3], input_dimention, self.filters, [1, 1], "SAME")


    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):

        block1 = self.conv1(inputs)
        block2 = self.conv2(block1)
        outputs = self.conv3(block2)
      
        outputs = outputs + self.shortcut(inputs)

        return tf.nn.relu(outputs)
