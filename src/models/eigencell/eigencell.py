import pickle as pk
import numpy as np
from itertools import product
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def get_metrics(test_y, pred):
    return (
        accuracy_score(test_y, pred),
        f1_score(test_y, pred, average="weighted"),
        confusion_matrix(test_y, pred),
    )


class EigenCell:
    def __init__(self, data, type="mono"):
        self.train_X, self.test_X, self.train_y, self.test_y = data
        self.le = LabelEncoder().fit(self.train_y)
        self.test_y = self.le.transform(self.test_y)
        self.k = 80

    def shift_mean(self):
        self.means = np.mean(self.train_X, axis=1)
        centred = self.train_X - self.means[:, np.newaxis]
        self.norms = np.linalg.norm(centred, np.inf, axis=1)
        self.norms[self.norms == 0] = 1
        self.train_X = centred / self.norms[:, np.newaxis]
        self.means = self.means.reshape((self.means.shape[0], -1))
        self.norms = self.norms.reshape((self.norms.shape[0], -1))

    def get_eigen(self):
        S = (1 / self.train_X.shape[1]) * (self.train_X.T @ self.train_X)
        self.D, C = np.linalg.eig(S)
        C = np.dot(self.train_X, C)
        self.C = C / np.linalg.norm(C, axis=0)

    def fit(self):
        self.shift_mean()
        self.get_eigen()

    def reduce_dimensionality(self):
        index = self.D.argsort()[::-1]
        D = self.D[index]
        C = self.C[:, index]
        query = (self.test_X - self.means.reshape(-1, 1)) / \
            self.norms.reshape(-1, 1)
        self.Z = C.T @ query
        self.Z[self.k:, :] = 0
        self.p = np.sum(D[: self.k]) / np.sum(D)

    def reconstruct_image(self):
        A = self.C @ self.Z
        A = A * self.norms.reshape(-1, 1)
        self.queries = A + self.means.reshape(-1, 1)

    def process_test_data(self):
        self.reduce_dimensionality()
        self.reconstruct_image()

    def nearest_image(self, query):
        query = query.reshape(query.shape[0], -1)
        norm = np.linalg.norm(query - self.train_X, axis=0)
        index = np.argmin(norm)
        return index

    def predict(self):
        self.process_test_data()
        indices = []
        for i in range(self.test_X.shape[1]):
            indices.append(self.nearest_image(self.queries[:, i]))

        pred = []
        for i in range(self.test_X.shape[1]):
            pred.append(self.train_y[indices[i]])

        self.pred = np.array(pred)
        self.pred = self.le.transform(self.pred)
        return self.pred

    def get_metrics(self):
        return get_metrics(self.pred, self.test_y)


