import pickle as pk
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# FILE_PATH = '../../../data/pickles'
FILE_PATH = './data/pickles'


def get_metrics(test_y, pred):
    return (
        accuracy_score(test_y, pred),
        f1_score(test_y, pred, average="weighted"),
        confusion_matrix(test_y, pred),
    )


class EigenCell:
    def __init__(self, type="mono"):
        self.get_data(type)
        self.le = LabelEncoder().fit(self.train_y)
        self.test_y = self.le.transform(self.test_y)
        self.k = 80

    def get_data(self, type):
        with open(FILE_PATH+f"/data_{type}.pkl", "rb") as f:
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

    def get_metrics(self):
        return get_metrics(self.pred, self.test_y)


if __name__ == "__main__":
    combinations = ["mono", "poly", "both"]
    results = dict()
    for combination in combinations:
        model = EigenCell(type=combination)
        with open(FILE_PATH+f"/model_{combination}.pkl", "wb") as f:
            pk.dump(model, f)
        # model.predict()
        # results[combination] = model.get_metrics()
        # print(f"Done processing: {combination}")
    # results_df = pd.DataFrame(results).T

    # with open(FILE_PATH+"/results_eigencell.pkl", "wb") as f:
        # pk.dump(results_df, f)
