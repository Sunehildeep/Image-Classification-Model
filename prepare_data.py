import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataPrep():
    def __init__(self, path):
        self.path = path
        self.categories = []

    def get_categories(self):
        print('Getting categories...')
        self.categories = os.listdir(self.path)
        return self.categories

    def prepare_data(self, split=(0.7, 0.15, 0.15), augment=False, batch_size=128):
        print('Preparing data...')
        if not self.categories:
            self.categories = self.get_categories()

        images = []
        labels = []

        print('Loading images...')
        for category in self.categories:
            path = os.path.join(self.path, category)
            for image in os.listdir(path):
                image_path = os.path.join(path, image)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0  # Normalize pixel values to [0, 1]

                images.append(img)
                # Convert category name to integer label
                labels.append(self.categories.index(category))

                if augment:
                    # Apply data augmentation here (e.g., rotation, flip, etc.)
                    img = cv2.flip(img, 1)
                    images.append(img)
                    labels.append(self.categories.index(category))

        # Shuffle the data
        data = list(zip(images, labels))
        np.random.shuffle(data)
        images, labels = zip(*data)

        # Split the data
        split_idx1 = int(len(images) * split[0])
        split_idx2 = split_idx1 + int(len(images) * split[1])
        train_data, val_data, test_data = images[:
                                                 split_idx1], images[split_idx1:split_idx2], images[split_idx2:]
        train_labels, val_labels, test_labels = labels[:
                                                       split_idx1], labels[split_idx1:split_idx2], labels[split_idx2:]

        train_data = np.array(train_data)
        val_data = np.array(val_data)
        test_data = np.array(test_data)
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)
        test_labels = np.array(test_labels)

        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_datagen = ImageDataGenerator(rescale=1.0/255.0)
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)

        train_generator = train_datagen.flow(
            train_data, train_labels, batch_size=batch_size)

        val_generator = val_datagen.flow(
            val_data, val_labels, batch_size=batch_size)

        test_generator = test_datagen.flow(
            test_data, test_labels, batch_size=batch_size)

        print('Data prepared.')
        return train_generator, val_generator, test_generator
