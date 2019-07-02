from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
import keras
from keras.optimizers import Adam


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=True):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


def transferlearning(regress=True):
    vgg16_model = keras.applications.vgg16.VGG16()
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    # pprint(model.summary())
    model.layers.pop()

    for layer in model.layers:
        layer.trainable = False

    if regress:
        model.add(Dense(2, activation='linear'))
    else:
        model.add(Dense(2, activation='softmax'))

    # pprint(model.summary())

    '''train the fine-tuned vgg16 model'''
    # model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
