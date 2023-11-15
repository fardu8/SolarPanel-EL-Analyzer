import os

import pandas as pd
import pickle as pk

from matplotlib import pyplot as plt
from sklearn import metrics

from src.data.preprocessing.mobilenetv2 import get_processed_data_loaders
from src.models.mobilenetv2.model_with_mobilenetv2_and_Kmeans import get_solar_cell_defect_detector
from src.models.mobilenetv2.testing import evaluate_detector_on_test_data
from src.visualization.visualize_results import plot_metrics

if __name__ == '__main__':

    metrics = dict()
    model_name = 'mobilenetv2'
    load_objects = True
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    metrics_df_filename = os.path.join(root_dir, 'data', 'pickles', 'results_mobilenetv2.pkl')

    train_loader, val_loader, test_loader , train_dataset, val_dataset = get_processed_data_loaders("both", load_objects)
    solar_cell_detector = get_solar_cell_defect_detector(train_loader, val_loader, train_dataset, val_dataset, 'data_both', load_weights=load_objects)
    metrics[("both", 224, "data")] = evaluate_detector_on_test_data(solar_cell_detector, test_loader, "data_both", load_results=load_objects)

    train_loader, val_loader, test_loader, train_dataset, val_dataset = get_processed_data_loaders("mono", load_objects)
    solar_cell_detector = get_solar_cell_defect_detector(train_loader, val_loader, train_dataset, val_dataset, 'data_mono', load_weights=load_objects)
    metrics[("mono", 224, "data")] = evaluate_detector_on_test_data(solar_cell_detector, test_loader, "data_mono", load_results=load_objects)

    train_loader, val_loader, test_loader, train_dataset, val_dataset = get_processed_data_loaders("poly", load_objects)
    solar_cell_detector = get_solar_cell_defect_detector(train_loader, val_loader, train_dataset, val_dataset, 'data_poly', load_weights=load_objects)
    metrics[("poly", 224, "data")] = evaluate_detector_on_test_data(solar_cell_detector, test_loader, "data_poly", load_results=load_objects)

    metrics_df = pd.DataFrame(metrics).T

    with open("data/pickles/results_mobilenetv2.pkl", "wb") as f:
        pk.dump(metrics_df, f)

    with open(f'data/pickles/results_{model_name}.pkl', 'rb') as f:
        data = pk.load(f)
    print(data)
    for index, row in data.iterrows():
        print(f'for {index[0]}')
        print('Accuracy: ', row[0])
        print('F1 score: ', row[1])

    # plot_metrics('mobilenetv2')


