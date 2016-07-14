from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

from elephas.spark_model import SparkMLlibModel
from elephas.utils.rdd_utils import to_labeled_point, lp_to_simple_rdd
from elephas import optimizers as elephas_optimizers

from pyspark import SparkContext, SparkConf

# Define basic parameters
batch_size = 64
nb_classes = 10
nb_epoch = 10

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(128, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# Create Spark context
conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[8]')
#sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
lp_rdd = to_labeled_point(sc, x_train, y_train, categorical=True)
rdd = lp_to_simple_rdd(lp_rdd, True, nb_classes)

# Initialize SparkModel from Keras model and Spark context
adagrad = elephas_optimizers.Adagrad()

spark_model = SparkMLlibModel(sc, model, optimizer=adagrad, frequency='batch', mode='asynchronous', num_workers=4)

# Train Spark model
spark_model.train(lp_rdd, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0,
                  validation_split=0.1, categorical=True, nb_classes=nb_classes)

# Evaluate Spark model by evaluating the underlying model
loss, acc = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', acc)
