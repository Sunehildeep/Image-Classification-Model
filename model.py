from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np


class Model():
    def __init__(self):
        print('Initializing model...')
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu',
                       input_shape=(224, 224, 3)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(22, activation='softmax'))

        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(
        ), metrics=[SparseCategoricalAccuracy()])
        self.model.summary()

    def train(self, train_generator, val_generator, epochs):
        # Check if a model.h5 file exists
        print('Checking for existing model...')
        model_exists = False
        try:
            self.model.load_weights('model.h5')
            model_exists = True
            print('Existing model found...')
        except:
            print('No existing model found...')
            pass

        # If a model exists, train it
        if model_exists:
            print('Training existing model...')
        else:
            print('Training new model...')

        checkpoint = ModelCheckpoint(
            'model.h5', monitor='val_loss', save_best_only=True, verbose=1)

        self.model.fit(train_generator, validation_data=val_generator,
                       epochs=epochs, callbacks=[checkpoint])

    def predict(self, image):
        print('Predicting...')
        image = cv2.imread(image)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Add a batch dimension

        return self.model.predict(image)

    def evaluate(self, test_generator):
        print('Evaluating...')
        self.model.evaluate(test_generator)

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        print('Saving model...')
        self.model.save_weights(path)
