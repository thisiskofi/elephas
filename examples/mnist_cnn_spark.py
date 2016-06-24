from __future__ import absolute_import
from __future__ import print_function

import keras.backend
keras.backend._keras_base_dir='./'

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers

from pyspark import SparkContext, SparkConf


# Define basic parameters
batch_size = 32
nb_classes = 10
nb_epoch = 10

# Create Spark context
conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[8]')
#sc = SparkContext(conf=conf)

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

from keras.constraints import maxnorm

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

keras_loss = 'categorical_crossentropy'
keras_optimizer = 'sgd'


# Compile model
model.compile(loss=keras_loss, optimizer=keras_optimizer, metrics=["accuracy"])

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, X_train, Y_train)

# Initialize SparkModel from Keras model and Spark context
print(model.to_yaml())
adagrad = elephas_optimizers.Adagrad(lr=0.01)
spark_model = SparkModel(sc,
                         model,
                         keras_losss=keras_loss,
                         keras_optimizer=keras_optimizer,
                         optimizer=adagrad,
                         frequency='batch',
                         mode='hogwild',
                         num_workers=2)

# Train Spark model
spark_model.train(rdd, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_split=0.1)

# Evaluate Spark model by evaluating the underlying model
loss, acc = spark_model.master_network.evaluate(X_test, Y_test, verbose=2)
