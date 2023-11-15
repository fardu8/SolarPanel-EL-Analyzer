import os

import joblib as joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def evaluate_detector_on_test_data(detector, test_loader, type, load_results=False):
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
    actual_file_name = os.path.join(root_dir, 'src', 'features', 'mobilenetv2', type, 'results',
                                    'actual.pkl')
    predictions_file_name = os.path.join(root_dir, 'src', 'features', 'mobilenetv2', type, 'results',
                                         'predictons.pkl')
    correct_predictions = 0
    total_predictions = 0
    predictions = []
    actual = []
    if load_results and os.path.exists(actual_file_name) and os.path.exists(predictions_file_name):
        actual = joblib.load(actual_file_name)
        predictions = joblib.load(predictions_file_name)
    else:
        for images, labels in test_loader:
            for i in range(images.size(0)):
                predicted_class = detector.detect_defect(images[i].unsqueeze(0))
                predictions.append(predicted_class)
                actual.append(labels[i])
                if predicted_class == labels[i].item():
                    correct_predictions += 1
                total_predictions += 1
        joblib.dump(actual, actual_file_name)
        joblib.dump(predictions, predictions_file_name)

    print(f"Accuracy on test data : {accuracy_score(actual, predictions) * 100}")

    return (
        accuracy_score(actual, predictions),
        f1_score(actual, predictions, average="weighted"),
        confusion_matrix(actual, predictions),
    )
