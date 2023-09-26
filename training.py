from prepare_data import DataPrep
from model import Model

if __name__ == "__main__":
    data_prep = DataPrep('data')
    train_generator, val_generator, test_generator = data_prep.prepare_data(
        batch_size=128)
    model = Model()
    model.train(train_generator, val_generator, epochs=20)
    model.save('model.h5')
    model.evaluate(test_generator)
