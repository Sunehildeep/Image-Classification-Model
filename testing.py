from model import Model
from prepare_data import DataPrep

if __name__ == "__main__":
    data_prep = DataPrep('data')
    categories = data_prep.get_categories()
    model = Model()
    model.load('model.h5')
    pred = model.predict('test.jpg')
    pred = pred.argmax()
    print(categories[pred])
