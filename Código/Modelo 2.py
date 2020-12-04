#Librerias necesarias para la implementación
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD

# Se cargan la lista de imágenes de entrenamiento y validación
folders_training = list(open('training_listaimagenes.txt'))
folders_testing = list(open('testing_listaimagenes.txt'))

# Unir la lista de imágenes con el path
path_file_training = []
for foldersito in folders_training:
    path = 'MTFL/'
    imagen = foldersito.strip()
    imagen = imagen.replace('\\', '/')
    path_file_training.append(os.path.join(path, imagen))

#Leer imágenes de entrenamiento
Pface_training = []
for image in path_file_training:
    #Reajustamos el tamaño de la imagen que es la permitida por la red neuronal en este caso (96,96)
    Pface_training.append(cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY), (96, 96), interpolation=cv2.INTER_AREA))
Pface_training = np.array(Pface_training)

#Carga el archivo csv con las marcas de las imágenes de entrenamiento
LMs_training = pd.read_csv('/data/estudiantes/kalau/MTFL/training.csv')
Spic_training = Pface_training.shape[1]
Xtraining = np.zeros((10000, Spic_training, Spic_training, 1))
Ytraining = np.zeros((10000, 10))

#Se determina el vector X y el vector Y de entrenamiento a partir de las imágenes y datos de entrenamiento
Xtraining[:, :, :, 0] = Pface_training[:, :, :] / 255.0
Ytraining[:, 0] = LMs_training.Lefteye_x / Spic_training
Ytraining[:, 1] = LMs_training.Lefteye_y / Spic_training
Ytraining[:, 2] = LMs_training.Righteye_x / Spic_training
Ytraining[:, 3] = LMs_training.Righteye_y / Spic_training
Ytraining[:, 4] = LMs_training.Nose_x / Spic_training
Ytraining[:, 5] = LMs_training.Nose_y / Spic_training
Ytraining[:, 6] = LMs_training.Leftmouth_x / Spic_training
Ytraining[:, 7] = LMs_training.Leftmouth_y / Spic_training
Ytraining[:, 8] = LMs_training.Rightmouth_x / Spic_training
Ytraining[:, 9] = LMs_training.Rightmouth_y / Spic_training

#Se ajustan las dimensiones de las marcas en el rostro
for ye in range(0, 4151):
    for yi in range(len(Ytraining[ye, :])):
        Ytraining[ye, yi] = (96.0 * Ytraining[ye, yi]) / 250.0

for ye in range(4151, 7650):
    for yi in range(len(Ytraining[ye, :])):
        Ytraining[ye, yi] = (96.0 * Ytraining[ye, yi]) / 400.0

for ye in range(7650, 10000):
    Ytraining[ye, 0::2] = (96.0 * Ytraining[ye, 0::2]) / 160.0
    Ytraining[ye, 1::2] = (96.0 * Ytraining[ye, 1::2]) / 216.0

#Unir la lista de imágenes con el path
path_file_testing = []
for foldersito in folders_testing:
    path = '/data/estudiantes/kalau/MTFL/'
    imagen = foldersito.strip()
    imagen = imagen.replace('\\', '/')
    path_file_testing.append(os.path.join(path, imagen))

#Leer imágenes de validación
Pface_testing = []
for image in path_file_testing:
    # Reajustamos el tamaño de la imagen que es la permitida por la red neuronal en este caso (96,96)
    Pface_testing.append(cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY), (96, 96), interpolation=cv2.INTER_AREA))
Pface_testing = np.array(Pface_testing)

#Carga el archivo csv con las marcas de las imágenes de validación
LMs_testing = pd.read_csv('/data/estudiantes/kalau/MTFL/testing.csv')
Spic_testing = Pface_testing.shape[1]
Xtesting = np.zeros((2995, Spic_testing, Spic_testing, 1))
Ytesting = np.zeros((2995, 10))

#Se determina el vector X y el vector Y de entrenamiento a partir de las imágenes y datos de validación
Xtesting[:, :, :, 0] = Pface_testing / 255.0
Ytesting[:, 0] = LMs_testing.Lefteye_x / Spic_testing
Ytesting[:, 1] = LMs_testing.Lefteye_y / Spic_testing
Ytesting[:, 2] = LMs_testing.Righteye_x / Spic_testing
Ytesting[:, 3] = LMs_testing.Righteye_y / Spic_testing
Ytesting[:, 4] = LMs_testing.Nose_x / Spic_testing
Ytesting[:, 5] = LMs_testing.Nose_y / Spic_testing
Ytesting[:, 6] = LMs_testing.Leftmouth_x / Spic_testing
Ytesting[:, 7] = LMs_testing.Leftmouth_y / Spic_testing
Ytesting[:, 8] = LMs_testing.Rightmouth_x / Spic_testing
Ytesting[:, 9] = LMs_testing.Rightmouth_y / Spic_testing

#Se ajustan las dimensiones de las marcas en el rostro
for ye in range(len(Ytesting)):
    for yi in range(len(Ytesting[ye, :])):
        Ytesting[ye, yi] = (96.0 * Ytesting[ye, yi]) / 250.0

#Implementación de la topología de la red neuronal convolucional
model = Sequential()
#Primer capa, 32 neuronas, el tamaño del filtro es (3,3), el tipo de activación y el tamaño de entrada
model.add(Conv2D(32, (3, 3), padding='same', activation='tanh', input_shape=(Spic_training, Spic_training, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))
#Capa de salida con las 5 marcas de la cara en x y ya
model.add(Dense(10, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

#Se entrena el modelo
model.fit(Xtraining, Ytraining, batch_size=128, epochs=200000, validation_data=(Xtesting, Ytesting), verbose=1)
#Guardar el modelo
model.save('model12.h5')