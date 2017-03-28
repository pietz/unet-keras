
# UNet (like) model for Keras with TF backend

from keras.models import Input, Model, Conv2D, Concatenate
from keras.models import MaxPooling2D, UpSampling2D, Dropout

def cdc_block(m, dim, acti, drop):
    m = Conv2D(dim, (3, 3), activation=acti, padding='same')(m)
    if drop != 0: m = Dropout(drop)(m)
    return Conv2D(dim, (3, 3), activation=acti, padding='same')(m)

def level_block(m, dim, depth, acti, drop):
    if depth > 0:
        n = cdc_block(m, dim, acti, drop)
        m = MaxPooling2D((2, 2))(n)
        m = level_block(m, 2*dim, depth-1, acti, drop)
        m = UpSampling2D((2, 2))(m)
        m = Concatenate(axis=1)([n, m])
    return cdc_block(m, dim, acti, drop)

def UNet(img_shape, start_dim=64, depth=4, acti='elu', drop=0):
    i = Input(input_shape=img_shape)
    o = level_block(i, start_dim, depth, acti, drop)
    return Model(input=i, output=o)