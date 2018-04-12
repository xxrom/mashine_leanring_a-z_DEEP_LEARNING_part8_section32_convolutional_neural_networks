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
  pool_size = (2, 2) # 2x2 рекомендованное значение чтобы не терять данные
))

# Adding a second convolutional layer
classifier.add(Convolution2D(
  32, # можно увеличить значение до 64
  3,
  3,
  border_mode = 'same'
  # input_shape = (64, 64, 3), # не нужно добавлять, keras сам разберется
  # так как уже была первая Convlutional2D
  activation = 'relu'
))
classifier.add(MaxPooling2D(
  pool_size = (2, 2) # 2x2 рекомендованное значение чтобы не терять данные
))

# Step 3 - Flattening # берем все Feature Maps и в векторз запихиваем =)
# если пропустить шаги 1,2 и сразу напрямую передать картинку, то
# у нас не будет информации об отдельныхчастях картинки, только об одной
classifier.add(Flatten()) # keras сам поймет, что нужно делать, на основе
  # предыдущего слоя

# Step - 4 Full Connection # создаем полно связанный граф
classifier.add(Dense(
  output_dim = 512, # 128, # experiment (около 100 надо брать 2 степени)
  activation = 'relu', # rectifier activation functon
))
  # output layer
classifier.add(Dense(
  output_dim = 1, # cat or dog
  activation = 'sigmoid', # sigmoid
))

# Compiling the CNN
classifier.compile(
  optimizer = 'adam', # метод оптимизации
  loss = 'binary_crossentropy', # так как только 2 класса могут быть (cadecorical_)
  metrics = ['accuracy'] # метод измерения качества модели
)

# Part 2 - Fitting the CNN to the images
# google => keras documentation => Preprocessing => ImageDataGenerator
# добавляет новые изображения, берет твои и модифицирует их как хочет =)
from keras.preprocessing.image import ImageDataGenerator

# чтобы не было переобучения, мы искуственно накручиваем себе новых
# изображений, которые будут случайно транформированы, будет выше точность
train_datagen = ImageDataGenerator( # randomly applying transformations
  rescale = 1./255, # пиксели 0 - 255, модифицируем к интервалу [0, 1]
  shear_range = 0.2, # shearing transformations (скосы, наклоны картинки)
  zoom_range = 0.2, # zoomes
  horizontal_flip = True) # перевернуть относительно горизонтали

# для проверки не нужно ничего трансормировать, только привести пиксели
# из вида [0, 255] к [0, 1]
test_datagen = ImageDataGenerator(rescale = 1./255)

# считываем изображения, преобразуем их к 64х64
training_set = train_datagen.flow_from_directory(
  'dataset/training_set', # file path to folder with images
  target_size = (64, 64), # размер входного изображения
  batch_size = 32, #32 количество случайных изображений, после которого
    # обновляем веса нейронки (коррекция весов)
  class_mode = 'binary') # количество классов 2 = binary!
# после выполнения => Found 8000 images belonging to 2 classes.

test_set = test_datagen.flow_from_directory(
  'dataset/test_set',  # file path to folder with images
  target_size = (64, 64), # размер входного изображения(первый слой сети)
  batch_size = 32, #32 количество случайных изображений, после которого
    # обновляем веса нейронки (коррекция весов)
  class_mode='binary') # количество классов 2 = binary!
# после выполнения => Found 2000 images belonging to 2 classes.

classifier.fit_generator(
  training_set, # набор изображений для тренеровок
  steps_per_epoch = 8000, # количество изображений для одной эпохи
  epochs = 25, # эпохи
  validation_data = test_set,# набор изображений для тестирования качества
  validation_steps = 2000) # количество тестовых изображений





















