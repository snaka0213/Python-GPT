import settings
from scripts.train import Train
#from scripts.predict import Predict
#from scripts.validate import Validate
from scripts.file_reader import FileReader
from scripts.inverted_index import InvertedIndex

def main():
    # train
    train = Train()
    train.load(settings.TrainFileName, settings.DEBUG)
    data_set, L = train._data_set, train.L
    i = InvertedIndex(L, data_set)
    i.write("test.json")
    i.open("test.json")

if __name__ == "__main__":
    main()
