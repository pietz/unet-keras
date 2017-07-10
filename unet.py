from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

'''
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first convolution
depth: zero indexed depth of the U-structure
inc_rate: rate at which the convolutional channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
bn: adds Batch Normalization if true
fcn: use strided convolutions instead of maxpooling if true
'''

def level_block(m, dim, depth, inc_rate, acti, dropout, bn, fcn):
    if depth > 0:
        n = Conv2D(dim, 3, activation=acti, padding='same')(m)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(dropout)(n) if dropout else n
        n = Conv2D(dim, 3, activation=acti, padding='same')(n)
        n = BatchNormalization()(n) if bn else n
        m = Conv2D(dim, 3, strides=2, activation=acti, padding='same')(n) if fcn else MaxPooling2D()(n)
        m = level_block(m, int(inc_rate*dim), depth-1, inc_rate, acti, dropout, bn, fcn)
        m = UpSampling2D()(m)
        m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        m = Concatenate(axis=3)([n, m])
    m = Conv2D(dim, 3, activation=acti, padding='same')(m)
    m = BatchNormalization()(m) if bn else m
    m = Conv2D(dim, 3, activation=acti, padding='same')(m)
    return BatchNormalization()(m) if bn else m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2, activation='relu', dropout=0.05, bn=False, fcn=False):
    i = Input(shape=img_shape)
    o = level_block(img_shape, start_ch, depth, inc_rate, activation, dropout, bn, fcn)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)
