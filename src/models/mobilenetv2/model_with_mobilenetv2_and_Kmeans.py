import numpy as np
import torch
from sklearn.cluster import KMeans

from src.models.mobilenetv2.model_architectures import Model1, FeatureExtractor, SolarCellDefectDetector
from src.models.mobilenetv2.training import get_trained_model2, get_trained_model1, get_trained_kmeans


def extract_features(data_loader, device, feature_extractor):
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = feature_extractor(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


def get_solar_cell_defect_detector(train_loader, val_loader, train_dataset, val_dataset, data_type, load_weights):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model1 = Model1(device, load_weights=load_weights)
    trained_model1 = get_trained_model1(model1, train_loader, val_loader, 0.001, 10, data_type)
    trained_model1.eval()

    feature_extractor = FeatureExtractor(trained_model1).to(device)
    train_features, train_labels = extract_features(train_loader, device, feature_extractor)
    val_features, val_labels = extract_features(val_loader, device, feature_extractor)
    K = 8

    kmeans, cluster_labels, cluster_val_labels = get_trained_kmeans(train_features, val_features, data_type, load_weights)

    trained_model2_instances = get_trained_model2(K, trained_model1, train_dataset, val_dataset,
                                                  cluster_labels, cluster_val_labels, device, 25, data_type, load_weights=load_weights)

    detector = SolarCellDefectDetector(trained_model2_instances, trained_model1, device, kmeans)
    return detector


