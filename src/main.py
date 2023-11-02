from src.data.preprocessing.mobilenetv2 import get_processed_data_loaders
from src.models.mobilenetv2.model_with_mobilenetv2_and_Kmeans import get_solar_cell_defect_detector
from src.models.mobilenetv2.testing import evaluate_detector_on_test_data

if __name__ == '__main__':

    train_loader, val_loader, test_loader, dataset, train_dataset, val_dataset, test_dataset = get_processed_data_loaders()
    solar_cell_detector = get_solar_cell_defect_detector(train_loader, val_loader, train_dataset, val_dataset)
    evaluate_detector_on_test_data(solar_cell_detector, test_loader)
