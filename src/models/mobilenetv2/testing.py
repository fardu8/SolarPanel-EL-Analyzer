import os

import joblib as joblib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def evaluate_detector_on_test_data(detector, test_loader, type, load_results=False):
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
    actual_file_name = os.path.join(root_dir, 'src', 'features', 'mobilenetv2', type, 'results',
                                    'actual.pkl')
    predictions_file_name = os.path.join(root_dir, 'src', 'features', 'mobilenetv2', type, 'results',
                                         'predictons.pkl')
    correct_classification_file_name = os.path.join(root_dir, 'src', 'features', 'mobilenetv2', type, 'results',
                                                    'correct_classification.pkl')
    incorrect_classification_file_name = os.path.join(root_dir, 'src', 'features', 'mobilenetv2', type, 'results',
                                                      'incorrect_classification.pkl')
    correct_predictions = 0
    total_predictions = 0
    predictions = []
    actual = []
    correct_classification = []
    incorrect_classification = []
    if load_results and os.path.exists(actual_file_name) and os.path.exists(predictions_file_name) and os.path.exists(
            correct_classification_file_name) and os.path.exists(incorrect_classification_file_name):
        actual = joblib.load(actual_file_name)
        predictions = joblib.load(predictions_file_name)
        correct_classification = joblib.load(correct_classification_file_name)
        incorrect_classification = joblib.load(incorrect_classification_file_name)
    else:
        for images, labels in test_loader:
            for i in range(images.size(0)):
                predicted_class = detector.detect_defect(images[i].unsqueeze(0))
                predictions.append(predicted_class)
                actual.append(labels[i])
                if predicted_class == labels[i].item():
                    correct_predictions += 1
                    correct_classification = (images[i].unsqueeze(0), labels[i].item())
                else:
                    incorrect_classification = (images[i].unsqueeze(0), labels[i].item())
                total_predictions += 1
        joblib.dump(actual, actual_file_name)
        joblib.dump(predictions, predictions_file_name)
        joblib.dump(correct_classification, correct_classification_file_name)
        joblib.dump(incorrect_classification, incorrect_classification_file_name)

    print(f"Accuracy on test data : {accuracy_score(actual, predictions) * 100}")

    return (
        accuracy_score(actual, predictions),
        f1_score(actual, predictions, average="weighted"),
        confusion_matrix(actual, predictions),
    ), correct_classification, incorrect_classification
