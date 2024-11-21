from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D


model = Sequential([
    DepthwiseConv2D(3, strides=1, padding='same', use_bias=False),
    # Remove any 'groups' argument if present in the original code
])

model.save("keras_model.h5")
