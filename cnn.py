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

# Initialising the CNN
classifier = Sequential() # init

# Step 1 - Convolution
classifier.add(Convolution2D(
    32, # feature detectores (default 32, next layer 128 or 256)
    3, # rows for feature detector
    3, # columns for feature detector
    border_mode = 'same', # default
    input_shape = (64, 64, 3), # imput photo format
      # expected format convert for one shape all photos
      # colored img = 3 array (rgb) / 3 arrays = (256 X 256)
      # black and white img = 1 array / 1 array = (256 X 256)
      # если использовать TensorFlow то сначала размер и потом 1D or 3D
        # input_shape = (256, 256, 3) если tensorFlow backend
        # thiana нужно будет (3, 256, 256)
    activation = 'relu' # rectifier activation function
    # убираем отрицательные пиксели, чтобы не было линейности в изображении
  ))

# Step 2 - Max Pooling (уменьшаем размер Feature Map другой матрицей 2х2)
# увеличиваем скорость работы и более устойчив к колебаниям и изменениям
# Feature Map получается после прохождения Feature Detector по изображению
# и получится Pooling Layer , который нужно будет в вектор превратить
# и потом передать в Full Connected Layers
classifier.add(MaxPooling2D(
    pool_size = (2, 2) # рекомендованное значение чтобы не терять данные
    ))



















