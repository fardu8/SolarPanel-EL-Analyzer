import pickle as pk
import numpy as np
from itertools import product
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class EigenCell:
    def __init__(self, data="data", dim=None, type="mono"):
        self.get_data(data, type, dim[0])
        self.le = LabelEncoder().fit(self.train_y)
        self.test_y = self.le.transform(self.test_y)
        self.process_test_data()

    def get_data(self, data, type, dim):
        with open(f"../data/pickles/{data}_{type}_{dim}.pkl", "rb") as f:
            (
                self.train_X,
                self.train_y,
                self.test_X,
                self.test_y,
                self.means,
                self.norms,
                self.k,
                self.C,
                self.D,
            ) = pk.load(f)

    def reduce_dimensionality(self):
        index = self.D.argsort()[::-1]
        D = self.D[index]
        C = self.C[:, index]
        query = (self.test_X - self.means.reshape(-1, 1)) / self.norms.reshape(-1, 1)
        self.Z = C.T @ query
        self.Z[self.k :, :] = 0
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
        indices = []
        for i in range(self.test_X.shape[1]):
            indices.append(self.nearest_image(self.queries[:, i]))

        pred = []
        for i in range(self.test_X.shape[1]):
            pred.append(self.train_y[indices[i]])

        self.pred = np.array(pred)
        self.pred = self.le.transform(self.pred)

    def get_metrics(self):
        return (
            accuracy_score(self.test_y, self.pred),
            f1_score(self.test_y, self.pred, average="weighted"),
            confusion_matrix(self.test_y, self.pred),
        )


if __name__ == "__main__":
    combinations = list(
        product(["mono", "poly", "both"], [224], ["data", "augmented"])
    )
    results = dict()
    for combination in combinations:
        type, num, data = combination
        model = EigenCell(data=data, dim=(num, num), type=type)
        model.predict()
        results[combination] = model.get_metrics()
        print(f"Done processing: {combination}")
    results_df = pd.DataFrame(results).T

    with open("../data/pickles/results.pkl", "wb") as f:
        pk.dump(results_df, f)
