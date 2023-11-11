import pickle as pk
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from itertools import product


class Image:
    def __init__(self, data="data", dim=None, type="mono"):
        self.get_data(data)
        self.set_dimension(dim)
        self.get_data_from_type(type)
        self.contrast_image()
        self.split_data()
        self.k = 80
        self.shift_mean()
        self.get_eigen()
        self.export_data(data, type, dim)

    def get_data(self, data):
        with open(f"../data/pickles/{data}.pkl", "rb") as f:
            self.images, self.probs, self.types = pk.load(f)

    def set_dimension(self, dimension):
        if dimension is None:
            self.dimension = (300, 300)
        else:
            self.dimension = dimension

    def get_data_from_type(self, type="mono"):
        if type == "both":
            return
        mask = self.types == type
        self.images = self.images[mask]
        self.probs = self.probs[mask]

    def min_max_normal(self, channel, c, d):
        O = (channel.astype("float") - c) * (255 / (d - c))
        O = np.clip(O, 0, 255)
        return O.astype("uint8")

    def HSV_contrast(self, I):
        R, G, B = cv2.split(I)

        H, S, V = cv2.split(cv2.cvtColor(I, cv2.COLOR_RGB2HSV))

        c = np.min(V)
        d = np.max(V)

        R = self.min_max_normal(channel=R, c=c, d=d)
        G = self.min_max_normal(channel=G, c=c, d=d)
        B = self.min_max_normal(channel=B, c=c, d=d)

        O = cv2.merge([R, G, B])
        O = cv2.cvtColor(O, cv2.COLOR_RGB2GRAY)

        return O

    def contrast_stretch(self, I):
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        I = self.HSV_contrast(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
        I = resize(I, (128, 128))

        gaus = cv2.GaussianBlur(I, (3, 3), 5.0)

        erode = cv2.erode(gaus, kernel=kernel).flatten()
        return erode

    def contrast_image(self):
        self.processed = []
        for I in self.images:
            self.processed.append(self.contrast_stretch(I))
        self.processed = np.stack(self.processed)

    def split_data(self):
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            self.processed, self.probs, test_size=0.25, stratify=self.probs
        )
        self.train_X, self.test_X = self.train_X.T, self.test_X.T

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

    def export_data(self, data, type, dim):
        with open(f"../data/pickles/{data}_{type}_{dim[0]}.pkl", "wb") as f:
            dump = (
                self.train_X,
                self.train_y,
                self.test_X,
                self.test_y,
                self.means,
                self.norms,
                self.k,
                self.C,
                self.D,
            )
            pk.dump(dump, f)


if __name__ == "__main__":
    combinations = list(product(["mono", "poly", "both"], [224], ["data", "augmented"]))
    for combination in combinations:
        type, num, data = combination
        processs = Image(data=data, dim=(num, num), type=type)
        print(f'Done processing: {combination}')
