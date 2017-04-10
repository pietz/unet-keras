
# U-Net model for Keras with TF backend

from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, UpSampling2D, Dropout

def conv_block(m, dim, activation, dropout):
    m = Conv2D(dim, (3, 3), activation=activation, padding='same')(m)
    if dropout != 0: m = Dropout(dropout)(m)
    return Conv2D(dim, (3, 3), activation=activation, padding='same')(m)

def level_block(m, dim, depth, acti, drop):
    if depth > 0:
        n = conv_block(m, dim, activation, dropout)
        m = MaxPooling2D((2, 2))(n)
        m = level_block(m, 2*dim, depth-1, activation, dropout)
        m = UpSampling2D((2, 2))(m)
        m = Concatenate(axis=1)([n, m])
    return conv_block(m, dim, activation, dropout)

def UNet(img_shape, start_dim=64, depth=4, activation='elu', dropout=0):
    i = Input(input_shape=img_shape)
    o = level_block(i, start_dim, depth, acti, drop)
    return Model(input=i, output=o)
