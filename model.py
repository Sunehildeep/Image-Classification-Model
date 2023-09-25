from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np

class Model():
    def __init__(self):
        print('Initializing model...')
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
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

        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss=SparseCategoricalCrossentropy(), metrics=[SparseCategoricalAccuracy()])
        self.model.summary()

    def train(self, train_data, train_labels, val_data, val_labels, epochs, batch_size=256):
        print('Training...')
        checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=1)
        
        # Create data generators for batch processing
        train_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize pixel values
        val_datagen = ImageDataGenerator(rescale=1.0/255.0)

        train_generator = train_datagen.flow(train_data, train_labels, batch_size=batch_size)
        val_generator = val_datagen.flow(val_data, val_labels, batch_size=batch_size)

        self.model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=[checkpoint])

    def predict(self, image):
        print('Predicting...')
        image = cv2.imread(image)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # Add a batch dimension

        return self.model.predict(image)
    
    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        print('Saving model...')
        self.model.save_weights(path)
