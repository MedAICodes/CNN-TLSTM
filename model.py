from tensorflow import keras
from tensorflow.keras import layers
from config import DROPOUT_RATE, TARGET_SHAPE

def residual_block_3d(inputs, filters, stride=1):
    x = layers.Conv3D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv3D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet3d(input_shape=(128, 128, 128, 1), num_classes=1, dropout_rate=DROPOUT_RATE):
    inputs = keras.Input(shape=input_shape)

    # Initial Conv
    x = layers.Conv3D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks
    filters = 64
    for _ in range(2):
        x = residual_block_3d(x, filters)

    filters = 128
    x = residual_block_3d(x, filters, stride=2)
    for _ in range(1):
        x = residual_block_3d(x, filters)

    filters = 256
    x = residual_block_3d(x, filters, stride=2)
    for _ in range(1):
        x = residual_block_3d(x, filters)

    filters = 512
    x = residual_block_3d(x, filters, stride=2)
    for _ in range(1):
        x = residual_block_3d(x, filters)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='ResNet3D')
    return model



