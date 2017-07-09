# U-Net for Keras

This is an implementation of the U-Net model for Keras. When using the default parameters this will be similar to the original architecture except for:

- Padded convolutions to keep image dimensions
- Elu instead of Relu
- No Dropout instead of 5%

U-Net: Convolutional Networks for Image Segmentation (https://arxiv.org/abs/1505.04597)

## Unscientific Thoughts

- Keeping the inc_rate at 1 often helped against overfitting and didn't hurt performance
- Dropout up to 0.2 yielded good results
- FCN feels like a good improvement over original U-Net
- I should really try separable and dilated convolutions
