import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def visualize_images_by_type_and_proba(images, proba, types):
    unique_proba = sorted(set(proba))
    unique_types = sorted(set(types))

    fig, axs = plt.subplots(len(unique_proba), len(unique_types), figsize=(10, 10))

    for i, prob in enumerate(unique_proba):
        for j, t in enumerate(unique_types):
            matching_indices = [idx for idx, (p, tp) in enumerate(zip(proba, types)) if p == prob and tp == t]
            if matching_indices:
                idx = matching_indices[0]
                axs[i, j].imshow(images[idx], cmap='gray')
                axs[i, j].set_title(f"Type: {t}\nProbability: {prob * 100:.0f}%")
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def show_metrics(data):
    print('Accuracy: ', data[0])
    print('F1 score: ', data[1])
    print("Confusion matrix: ")
    metrics.ConfusionMatrixDisplay(data[2]).plot(values_format='.0f')
    plt.show()


def display_cluster_samples(cluster_labels, images, num_samples=5):
    unique_labels = np.unique(cluster_labels)
    fig, axes = plt.subplots(len(unique_labels), num_samples, figsize=(15, len(unique_labels) * 3))

    for cluster_id, ax in zip(unique_labels, axes):
        # Get indices of images belonging to this cluster
        idxs = np.where(cluster_labels == cluster_id)[0]
        # Randomly sample images from this cluster
        sample_idxs = np.random.choice(idxs, num_samples, replace=False)

        for i, sample_idx in enumerate(sample_idxs):
            ax[i].imshow(images[sample_idx].squeeze(), cmap='gray')
            ax[i].axis('off')
            if i == 0:
                ax[i].set_title(f'Cluster {cluster_id}')

    plt.tight_layout()
    plt.show()
