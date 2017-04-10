
# U-Net model for Keras with TF backend

from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, Activation

def conv_block(m, dim, acti, drop):
    m = Conv2D(dim, (3, 3), activation=acti, padding='same')(m)
    if drop != 0: m = Dropout(drop)(m)
    return Conv2D(dim, (3, 3), activation=acti, padding='same')(m)

def level_block(m, dim, depth, acti, drop):
    if depth > 0:
        n = conv_block(m, dim, acti, drop)
        m = MaxPooling2D((2, 2))(n)
        m = level_block(m, 2*dim, depth-1, acti, drop)
        m = UpSampling2D((2, 2))(m)
        m = Concatenate(axis=3)([n, m])
    return conv_block(m, dim, acti, drop)

def UNet(img_shape, n_out=1, dim=64, depth=4, acti='elu', drop=0, flatten=False):
    i = Input(shape=img_shape)
    o = level_block(i, dim, depth, acti, drop)
    o = Conv2D(n_out, (1, 1))(o)
    if flatten:
        o = Reshape(n_out, img_shape[0] * img_shape[1])(o)
        o = Permute((2, 1))(o)
    o = Activation('softmax')(o)
    return Model(inputs=i, outputs=o)
