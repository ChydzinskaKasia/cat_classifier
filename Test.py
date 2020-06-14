print("Trenowanie sieci ehhhhhhhh")
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from PIL.Image import core as _imaging
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import Sequential  # inicjalizacja sieci neuronowej
from tensorflow.python.keras.layers import Conv2D, GlobalAveragePooling2D  # operacja splotu 2D w sieci
from tensorflow.python.keras.layers import MaxPooling2D  # operacja maxpoolingu 2D w sieci
from tensorflow.python.keras.layers import Flatten  # wyrównanie do kolumny w sieci
from tensorflow.python.keras.layers import Dense  # przejście do klasycznej ANN
from tensorflow.python.keras.layers import Dropout

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

##Sciezki do zdjec
Base_path = os.path.dirname(os.path.abspath(__file__))
Train_path = os.path.join(Base_path, r'Baza\\Training')
TrainPieski_path = os.path.join(Base_path, r'Baza\\Training\\Pieski')
TrainKotki_path = os.path.join(Base_path, r'Baza\\Training\\Kotki')
Test_path = os.path.join(Base_path, r'Baza\\Test')
TestPieski_path = os.path.join(Base_path, r'Baza\\Test\\Pieski')
TestKotki_path = os.path.join(Base_path, r'Baza\\Test\\Kotki')

##Obliczenie rozmiaru zestawu do uczenia i treningowego
size_test_Pieski = len(os.listdir(TestPieski_path))
size_test_Kotki = len(os.listdir(TestKotki_path))
size_test = size_test_Pieski + size_test_Kotki
size_Kotki = len(os.listdir(TrainKotki_path))
size_Pieski = len(os.listdir(TrainPieski_path))
size_train = size_Kotki + size_Pieski


##Funkcja do zmiany rozmiaru zdjec
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = img_to_array(target_size)
    image = np.expand_dims(image, axis=0)
    return image


##Funkcja glowna
if __name__ == '__main__':
    BATCH_SIZE = 60
    EPOCHS = 10
    IMG_HEIGHT = 200
    IMG_WIDTH = 200

    # print(tf.__version__)

    # 1./255 zmienia format z uint8 do float32 w zakresie [0,1].
    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # Wczytywanie bazy ze zdjeciami do treningu
    train_data_gen = train_image_generator.flow_from_directory(directory=str(Train_path),
                                                               batch_size=BATCH_SIZE,
                                                               shuffle=True,
                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               class_mode='categorical')
    # Wczytywanie bazy ze zdjeciami do testowania
    test_data_gen = train_image_generator.flow_from_directory(directory=str(Test_path),
                                                              batch_size=BATCH_SIZE,
                                                              shuffle=True,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

    # Kolejne warstwy sieci
    classifier = Sequential()

    # Add 2 convolution layers
    classifier.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(200, 200, 3), activation='relu'))
    #classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

    # Add pooling layer
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Add dropout
    classifier.add(Dropout(0.2))
    # Add 2 more convolution layers
    classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    #classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

    # Add max pooling layer
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Add dropout
    classifier.add(Dropout(0.2))

    # Add 2 more convolution layers
    classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    #classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

    # Add max pooling layer
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Add dropout
    #classifier.add(Dropout(0.2))

    # Add global average pooling layer
    #classifier.add(GlobalAveragePooling2D())

    # Add Flatten
    classifier.add(Flatten())

    # Add full connection
    classifier.add(Dense(units=2, activation='softmax'))
    #classifier.add(Dense(1, activation='sigmoid'))
    # img = np.expand_dims(ndimage.imread)

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"]),

    classifier.summary(),

    history = classifier.fit_generator(
        train_data_gen,
        steps_per_epoch=size_train // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=test_data_gen,
        validation_steps=size_test // BATCH_SIZE
    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Skuteczność treningowa')
plt.plot(epochs_range, val_acc, label='Skuteczność testowa')
plt.legend(loc='lower right')
plt.title('Skuteczność treningowa i testowa')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Straty podczas treningu')
plt.plot(epochs_range, val_loss, label='Straty podczas testu')
plt.legend(loc='upper right')
plt.title('Straty podczas treningu i testu')
plt.show()

# Zapisywanie wytrenowanego modelu
savepath1 = os.path.join(Base_path, r'PiesvsKot_model.h5')
savepath2 = os.path.join(Base_path, r'PiesvsKot_model_weights.h5')
tf.keras.models.save_model(
    classifier, savepath1
)
classifier.save_weights(savepath2)

print("Model został zapisany w folderze głównym katalogu projektu")
