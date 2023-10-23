from src.data.elpv_reader import load_dataset
from visualization.visualize import visualize_images_by_type_and_proba

if __name__ == '__main__':
    images, proba, types = load_dataset()
    visualize_images_by_type_and_proba(images, proba, types)
