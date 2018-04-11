# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# здесь не будет Data Preprocessing как в ANN, так как данные уже разбиты
# на категории для обучения и для тестирования нейронки
# поэтому сразу приступаем разрабатывать CNN
# Part 1 - Building the CNN
from keras.models import Sequential # for init NN для слоев нейронки
from keras.layers import Convolution2D # для первого слоя CNN (2D - imges)
from keras.layers import MaxPooling2D # for pooling layers (2 step)
from keras.layers import Flatten # for flattening (3 step)
from keras.layers import Dense # add fully connected layer in classic ANN

