def evaluate_detector_on_test_data(detector, test_loader):
    correct_predictions = 0
    total_predictions = 0

    for images, labels in test_loader:
        for i in range(images.size(0)):
            predicted_class = detector.detect_defect(images[i].unsqueeze(0))
            if predicted_class == labels[i].item():
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy on test data : {accuracy}")