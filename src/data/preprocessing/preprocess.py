import pickle as pk
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from skimage.transform import resize

FILE_PATH = './data/pickles'


class Preprocess:
    def __init__(self):
        self.get_data()
        self.set_dimension()
        self.contrast_image()
        self.split_data()
        self.k = 80
        self.shift_mean()
        self.get_eigen()
        self.export_data()

    def get_data(self):
        with open(FILE_PATH+"/data.pkl", "rb") as f:
            self.images, self.probs, self.types = pk.load(f)

    def set_dimension(self):
        self.dimension = 128

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
        processed = []
        for I in self.images:
            processed.append(self.contrast_stretch(I))
        self.processed = np.stack(processed)

    def split_data(self):
        y = np.c_[self.probs, self.types]
        self.train_X, self.test_X, self.train_y, test_y = train_test_split(
            self.processed, y, test_size=0.25, stratify=y, random_state=1
        )
        self.train_y = self.train_y[:, 0]

        self.train_X, self.test_X = self.train_X.T, self.test_X.T
        self.test_y = test_y[:, 0]
        self.types = test_y[:, 1]

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

    def export_data(self):
        with open(FILE_PATH+"/preprocessed.pkl", "wb") as f:
            dump = (
                self.train_X,
                self.train_y,
                self.test_X,
                self.test_y,
                self.types,
                self.means,
                self.norms,
                self.k,
                self.C,
                self.D,
            )
            pk.dump(dump, f)


if __name__ == "__main__":
    processs = Preprocess()
