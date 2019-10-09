from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model


def lenet5(input_shape, num_classes=10):
    net = {}
    input_tensor = Input(shape=input_shape)
    net['input'] = input_tensor

    #c1
    net['c1'] = Convolution2D(6, (5, 5),
                              activation='tanh',
                              name='c1')(net['input'])

    #s2
    net['s2'] = AveragePooling2D((2, 2), strides=2,
                                 name='s2')(net['c1'])

    #c3
    net['c3'] = Convolution2D(16, (5, 5),
                              activation='tanh',
                              name='c3')(net['s2'])

    #s4
    net['s4'] = AveragePooling2D((2, 2), strides=2,
                                 name='s4')(net['c3'])

    #c5
    net['c5'] = Convolution2D(120, (5, 5),
                              activation='tanh',
                              name='c5')(net['s4'])

    #f6
    net['flat'] = Flatten()(net['c5'])
    net['f6'] = Dense(84, activation='tanh',
                      name='f6')(net['flat'])

    #output
    net['output'] = Dense(num_classes, activation='softmax',
                          name='output')(net['f6'])

    lenet5 = Model(net['input'], net['output'])
    return lenet5