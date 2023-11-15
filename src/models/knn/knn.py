import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm.auto import tqdm 
import pickle as pk
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, f1_score

def evaluate_results(results):
    true_labels = []
    for key, value in results[2].items():
        true_labels.extend([key] * value[1])

    overall_accuracy = accuracy_score(true_labels, results[3])
    conf_matrix = confusion_matrix(true_labels, results[3])
    f1 = f1_score(true_labels, results[3], average='weighted')

    return overall_accuracy, f1, conf_matrix

class ImageClassifier:
    def __init__(self):
        pass

    @staticmethod
    def get_ith(img, c):
        itr = 0
        ith = 0
        i = 0

        while i < len(c):
            if i == 0:
                itr = np.linalg.norm(np.subtract(np.array(img), np.array(c[i])))
            else:
                dist = np.linalg.norm(np.subtract(np.array(img), np.array(c[i])))
                if dist < itr:
                    ith = i
                    itr = dist
            i += 1

        return ith

    def classifier(self, x, y):
        feat = {}

        for key, value in tqdm(x.items()):
            class_of_x = []
            for img in value:
                hist = np.zeros(len(y))

                if img is not None and all(feature is not None for feature in img):
                    for each_feature in img:
                        ind = self.get_ith(each_feature, y)
                        hist[ind] += 1

                class_of_x.append(hist)
            feat[key] = class_of_x

        return feat
    
class KNNClassifier:
    def __init__(self, train_data, k=5):
        self.train_data = train_data
        self.k = k

    def predict(self, test_data):
        true_classification = 0
        c = {}
        keys = list(test_data.keys())
        i = 0
        total = 0
        p = []
        while i < len(keys):
            key_i = keys[i]
            c[key_i] = [0, 0]
            j = 0

            while j < len(test_data[key_i]):
                tst = test_data[key_i][j]
                ns = []
                keys_t = list(self.train_data.keys())
                k = 0
                while k < len(keys_t):
                    m = 0
                    while m < len(self.train_data[keys_t[k]]):
                        train = self.train_data[keys_t[k]][m]
                        dist = distance.euclidean(tst, train)
                        ns.append((dist, keys_t[k]))
                        m += 1

                    k += 1
                ns.sort(key=lambda x: x[0])
                k_ns = ns[:self.k]

                votes = {}
                n = 0
                while n < len(k_ns):
                    neighbor = k_ns[n]
                    votes[neighbor[1]] = votes.get(neighbor[1], 0) + 1
                    n += 1

                p.append(max(votes, key=votes.get))
                if key_i == max(votes, key=votes.get):
                    true_classification += 1
                    c[key_i][0] += 1

                total += 1
                c[key_i][1] += 1

                j += 1

            i += 1

        return [total, true_classification, c, p]

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pk.dump(self, file)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            model = pk.load(file)
        return model




def create_sift_feature_database(X_train, Y_train_prob, Y_train_type):
    sift = cv2.SIFT_create()
    desc = []
    sift_database = {
        'fully_functional': [],
        'possibly_defective': [],
        'likely_defective': [],
        'certainly_defective': []
    }

    for img, p, t in tqdm(zip(X_train, Y_train_prob, Y_train_type), total=len(X_train)):
        kp, des = sift.detectAndCompute(img, None)
        
        if des is not None:
            desc.extend(des)
        
        if p >= 0.99:
            img_class = 'certainly_defective'
        elif p >= 0.66:
            img_class = 'likely_defective'
        elif p >= 0.30:
            img_class = 'possibly_defective'
        else:
            img_class = 'fully_functional'

        sift_database[img_class].append(( des))

    return [desc,sift_database]

def K_Means_Clustering(k, des):
    K_Means_Clustering = KMeans(n_clusters = k, n_init=10)
    K_Means_Clustering.fit(des)
    k_means_clusters = K_Means_Clustering.cluster_centers_ 
    return k_means_clusters

with open('../../../data/pickles/data.pkl', 'rb') as f:
    images, proba, types = pk.load(f)
    
_images = []
_proba = []
_types = []

for img, prob, typ in zip(images, proba, types):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(img)
    cs_img = cv2.convertScaleAbs(img, alpha=255.0/(max_val - min_val), beta=-min_val * 255.0/(max_val - min_val))
    b_image = cv2.GaussianBlur(cs_img, (5, 5), 0)
    
    _images.extend([img,b_image])
    _proba.extend([prob] * 2)
    _types.extend([typ] * 2)

images = _images
proba = _proba
types = _types


labels = [f'{p}_{t}' for p, t in zip(proba, types)]

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=42, stratify=labels)
Y_train_prob, Y_train_type = zip(*[label.split("_") for label in y_train])
Y_test_prob, Y_test_type = zip(*[label.split("_") for label in y_test])

Y_train_prob = [float(prob) for prob in Y_train_prob]
Y_test_prob = [float(prob) for prob in Y_test_prob]

sift = cv2.SIFT_create()
img_1 = X_train[0].copy()
img_2 = X_train[20].copy()

keypoints, descriptors = sift.detectAndCompute(img_1, None)
img_1_kp = cv2.drawKeypoints(img_1, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoints, descriptors = sift.detectAndCompute(img_2, None)
img_2_kp = cv2.drawKeypoints(img_2, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(12, 6))
plt.subplot(141), plt.imshow(img_1, cmap='gray'), plt.title('Image')
plt.subplot(142), plt.imshow(img_1_kp), plt.title('Keypopints')
plt.subplot(143), plt.imshow(img_2, cmap='gray'), plt.title('Image')
plt.subplot(144), plt.imshow(img_2_kp), plt.title('Keypopints')
plt.savefig('../../../plots/sift_eda.png')
plt.close()

sift_features = create_sift_feature_database(X_train, Y_train_prob, Y_train_type)

list_of_desc = sift_features[0] 
train_features = sift_features[1]

X_test_mono = []
X_test_poly = []
Y_test_prob_mono = []
Y_test_prob_poly = []
Y_test_type_mono = []
Y_test_type_poly = []
for i in range(len(X_test)):
    if Y_test_type[i] == 'mono':
        X_test_mono.append(X_test[i])
        Y_test_prob_mono.append(Y_test_prob[i])
        Y_test_type_mono.append(Y_test_type[i])
    else:
        X_test_poly.append(X_test[i])
        Y_test_prob_poly.append(Y_test_prob[i])
        Y_test_type_poly.append(Y_test_type[i])
        
        
test_features = create_sift_feature_database(X_test,Y_test_prob,Y_test_type)[1] 
test_features_mono = create_sift_feature_database(X_test_mono,Y_test_prob_mono,Y_test_type_mono)[1] 
test_features_poly = create_sift_feature_database(X_test_poly,Y_test_prob_poly,Y_test_type_poly)[1]

Y = (K_Means_Clustering(200, list_of_desc))
classifier = ImageClassifier()

train = classifier.classifier(train_features, Y) 
test = classifier.classifier(test_features, Y)
test_mono = classifier.classifier(test_features_mono, Y)
test_poly = classifier.classifier(test_features_poly, Y)

knn_classifier = KNNClassifier(train, k=5)
knn_classifier.save_model('../../features/knn/knn_model.pkl')
results = knn_classifier.predict(test)
results_mono = knn_classifier.predict(test_mono)
results_poly = knn_classifier.predict(test_poly)

metrics = {}
metrics[('mono', 'data')]  = list(evaluate_results(results))
metrics[('poly', 'data')]  = list(evaluate_results(results_mono))
metrics[('both', 'data')]  = list(evaluate_results(results_poly))

metrics = pd.DataFrame(metrics).T

with open("../../../data/pickles/results_knn.pkl", "wb") as f:
        pk.dump(metrics, f)