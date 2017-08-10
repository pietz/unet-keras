# U-Net for Keras

This is an implementation of the U-Net model for Keras. When using the default parameters this will be the same as the original architecture, except that this code will use padded convolutions. Also, the original paper does not state the amount of Dropout used. Since 0.5 is a common choice in the industry, it was used as the default value. If you are aware of other differences, please contact me.

U-Net: Convolutional Networks for Image Segmentation (https://arxiv.org/abs/1505.04597)
