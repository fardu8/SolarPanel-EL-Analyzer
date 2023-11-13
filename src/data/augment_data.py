from elpv_reader import load_dataset
import numpy as np
import pickle as pk


def augment_data(data):
    images, probs, types = data
    aug_im, aug_prob, aug_types = [], [], []

    for image, prob, type in zip(images, probs, types):
        aug_im.extend([image, image[:, ::-1], image[::-1, :], image[::-1, ::-1]])
        aug_prob.extend([prob for _ in range(4)])
        aug_types.extend([type for _ in range(4)])
    aug_im = np.array(aug_im)
    aug_prob = np.array(aug_prob)
    aug_types = np.array(aug_types)

    return aug_im, aug_prob, aug_types


if __name__ == "__main__":
    data = load_dataset()
    with open("../../data/pickles/data.pkl", "wb") as f:
        pk.dump(data, f)
    augmented = augment_data(data)
    images, proba, types = augmented
    with open("../../data/pickles/augmented.pkl", "wb") as f:
        pk.dump(augmented, f)