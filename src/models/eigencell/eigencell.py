import pickle as pk
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

FILE_PATH = './data/pickles'


def get_metrics(test_y, pred):
    return (
        accuracy_score(test_y, pred),
        f1_score(test_y, pred, average="weighted"),
        confusion_matrix(test_y, pred),
    )


