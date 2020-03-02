import tensorflow as tf
import numpy as np
import os
import sys
import MyLayer

class Model(tf.keras.Model):
    def __init__(self, opt_name, alpha, lambd, BATCH_SIZE, train_acc, test_acc):
        super(Model, self).__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.BATCH_SIZE = BATCH_SIZE
        self.loss_object  = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.train_acc = train_acc
        self.test_acc  = test_acc
        
        self.conv1   = tf.keras.layers.TimeDistributed(MyLayer.ConvLayer([3, 3], 3, 16, [2, 2], "VALID"))
        self.pool1    = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D())
        self.resnet1 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(32, [2, 2]))
        self.resnet2 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(32, [1, 1]))
        self.resnet3 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(32, [1, 1]))
        self.resnet4 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(64, [2, 2]))
        self.resnet5 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(64, [1, 1]))
        self.resnet6 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(64, [1, 1]))
        self.resnet7 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(128, [2, 2]))
        self.resnet8 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(128, [1, 1]))
        self.resnet9 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(128, [1, 1]))
        self.resnet10 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(256, [2, 2]))
        self.resnet11 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(256, [1, 1]))
        self.resnet12 = tf.keras.layers.TimeDistributed(MyLayer.ResNetLayer(256, [1, 1]))

        self.flatten1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.pool2    = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalMaxPool2D())
        self.lstm      = tf.keras.layers.LSTM(1024, time_major=False, activation='tanh')
        self.dense   = tf.keras.layers.Dense(64, activation='softmax')
        
        if opt_name=="Adam":
            self.opt             = tf.keras.optimizers.Adam(self.alpha)
        if opt_name=="Sgd":
            self.opt             = tf.keras.optimizers.SGD(self.alpha)

    def call(self, inputs, mask, training=None):
        #conv_layerに対するoutputsの形状(batchsize, frames, height, weight, channels)
        outputs = self.conv1(inputs,mask=mask)
        outputs = self.pool1(outputs,mask=mask)
        outputs = self.resnet1(outputs,mask=mask)
        outputs = self.resnet2(outputs,mask=mask)
        outputs = self.resnet3(outputs,mask=mask)
        outputs = self.resnet4(outputs,mask=mask)
        outputs = self.resnet5(outputs,mask=mask)
        outputs = self.resnet6(outputs,mask=mask)
        outputs = self.resnet7(outputs,mask=mask)
        outputs = self.resnet8(outputs,mask=mask)
        outputs = self.resnet9(outputs,mask=mask)
        outputs = self.resnet10(outputs,mask=mask)
        outputs = self.resnet11(outputs,mask=mask)
        outputs = self.resnet12(outputs,mask=mask)

        outputs = self.pool2(outputs,mask=mask)
        #reshapeに対するoutputsの形状(batchsize, framges, height*weight*channels)
        outputs = self.flatten1(outputs,mask=mask)
        outputs = self.lstm(outputs, mask=mask)
        #dense_layerに対する出力(batchsize, class)
        outputs = self.dense(outputs)   
        return outputs

    def train_step(self, images, labels, mask):
        with tf.GradientTape() as tape:
            pred = self(images, mask, True)
            loss  = self.loss_object(labels, pred)

        grads   = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        self.train_acc.update_state(labels, pred)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=self.BATCH_SIZE)
        return loss
    
    def test_step(self, images, labels, mask):
        pred = self(images, mask, False)
        loss =  self.loss_object(labels, pred)
        loss =  tf.nn.compute_average_loss(loss, global_batch_size=self.BATCH_SIZE)
        self.test_acc.update_state(labels, pred)

        return loss

    #仕様書によると、experimental_run_v2は@tf.function内に書かないといけないみたい。
    @tf.function
    def distributed_train_step(self, images, labels, mask):
            from main import mirrored_strategy#この汚い部分は、後で修正
            return  mirrored_strategy.experimental_run_v2(self.train_step, args=(images, labels, mask))

    @tf.function
    def distributed_test_step(self, images, labels, mask):
            from main import mirrored_strategy
            return  mirrored_strategy.experimental_run_v2(self.test_step,  args=(images, labels, mask))

    def accuracy_reset(self):
        train_accuracy.reset_states()
        test_accuracy.reset_states()
    
  