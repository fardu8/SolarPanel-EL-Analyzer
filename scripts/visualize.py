import matplotlib.pyplot as plt


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
