from prepare_data import DataPrep
from model import Model

if __name__ == "__main__":
    data_prep = DataPrep('data')
    train_data, train_labels, val_data, val_labels = data_prep.prepare_data()
    model = Model()
    model.train(train_data, train_labels, val_data, val_labels, 10)
    model.save('model.h5')