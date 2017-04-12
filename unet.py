from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Reshape
from keras.layers import UpSampling2D, Dropout, Activation, Permute

def level_block(m, dim, depth, acti):
    if depth > 0:
        n = Conv2D(dim, (3, 3), activation=acti, padding='same')(m)
        n = Conv2D(dim, (3, 3), activation=acti, padding='same')(n)
        m = MaxPooling2D((2, 2))(n)
        m = level_block(m, 2*dim, depth-1, acti, drop)
        m = UpSampling2D((2, 2))(m)
        m = Conv2D(dim, (2, 2), activation=acti, padding='same')(m)
        m = Concatenate(axis=3)([n, m])
    m = Conv2D(dim, (3, 3), activation=acti, padding='same')(m)
    return Conv2D(dim, (3, 3), activation=acti, padding='same')(m)

def UNet(img_shape, n_out=1, dim=64, depth=4, acti='relu', flatten=False):
    i = Input(shape=img_shape)
    o = level_block(i, dim, depth, acti, drop)
    o = Conv2D(n_out, (1, 1))(o)
    if flatten:
        o = Reshape(n_out, img_shape[0] * img_shape[1])(o)
        o = Permute((2, 1))(o)
    o = Activation('softmax')(o)
    return Model(inputs=i, outputs=o)
