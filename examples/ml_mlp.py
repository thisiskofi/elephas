from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

from elephas.ml_model import ElephasEstimator
from elephas.ml.adapter import to_data_frame
from elephas import optimizers as elephas_optimizers

from pyspark import SparkContext, SparkConf
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline


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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create Spark context
conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[8]')
#sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
df = to_data_frame(sc, x_train, y_train, categorical=True)
test_df = to_data_frame(sc, x_test, y_test, categorical=True)

# Define elephas optimizer
adagrad = elephas_optimizers.Adagrad()

# Initialize Spark ML Estimator
estimator = ElephasEstimator()
estimator.set_keras_model_config(model.to_yaml())
estimator.set_optimizer_config(adagrad.get_config())
estimator.set_nb_epoch(nb_epoch)
estimator.set_batch_size(batch_size)
estimator.set_num_workers(4)
estimator.set_verbosity(2)
estimator.set_validation_split(0.1)
estimator.set_categorical_labels(True)
estimator.set_nb_classes(nb_classes)

estimator.set_frequency('batch')

# Fitting a model returns a Transformer
pipeline = Pipeline(stages=[estimator])
fitted_pipeline = pipeline.fit(df)

# Evaluate Spark model by evaluating the underlying model
prediction = fitted_pipeline.transform(test_df)
pnl = prediction.select("label", "prediction")
pnl.show(100)

prediction_and_label = pnl.map(lambda row: (row.label, row.prediction))
metrics = MulticlassMetrics(prediction_and_label)
print("Precision:", metrics.precision())
print("Recall:", metrics.recall())
