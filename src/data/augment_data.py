from elpv_reader import load_dataset
import pickle as pk

FILE_PATH = './data/pickles'
if __name__ == "__main__":
    data = load_dataset()
    with open(FILE_PATH + "/data.pkl", "wb") as f:
        pk.dump(data, f)
