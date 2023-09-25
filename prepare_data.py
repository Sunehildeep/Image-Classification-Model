import os
import cv2
import numpy as np

# Class for data preparation
class DataPrep():
    def __init__(self, path):
        self.path = path
        self.categories = []

    def get_categories(self):
        print('Getting categories...')
        self.categories = os.listdir(self.path)
        return self.categories
    
    def prepare_data(self, split=0.8):
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
                labels.append(self.categories.index(category))  # Convert category name to integer label

        # Shuffle the data
        data = list(zip(images, labels))
        np.random.shuffle(data)
        images, labels = zip(*data)

        # Split the data
        split_idx = int(len(images) * split)
        train_data, val_data = images[:split_idx], images[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]

        print('Data prepared.')
        return np.array(train_data), np.array(train_labels), np.array(val_data), np.array(val_labels)
