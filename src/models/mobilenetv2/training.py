import os

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader

from src.models.mobilenetv2.model_architectures import Model2


def get_trained_model1(model1, train_loader, val_loader, learning_rate, epochs):
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
    file_name = os.path.join(root_dir, 'src', 'features', 'mobilenetv2', 'model1',
                             'model_1_weights.pth')
    if os.path.exists(file_name) and model1.load_weights:
        model1.model.load_state_dict(torch.load(file_name))
        model1.model.eval()
        print("Weights loaded for model 1 from the file: ", file_name)
        return model1.model
    else:
        model1.model.load_state_dict(torch.load(file_name))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model1.model.parameters(), lr=learning_rate)

        num_epochs = epochs

        model = model1.model.to(model1.device)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            for images, labels in train_loader:
                images, labels = images.to(model1.device), labels.to(model1.device)

                optimizer.zero_grad()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            train_accuracy = correct_predictions / total_predictions * 100
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}, Training Accuracy: {train_accuracy:.2f}%")

            model.eval()
            total_val_loss = 0.0
            correct_val_predictions = 0
            total_val_predictions = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(model1.device), labels.to(model1.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct_val_predictions += (predicted == labels).sum().item()
                    total_val_predictions += labels.size(0)

            val_accuracy = correct_val_predictions / total_val_predictions * 100
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {total_val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy:.2f}%")

        torch.save(model.state_dict(), file_name)
        model1.model = model
        return model


def get_trained_model2(K, model, train_dataset, val_dataset, cluster_labels, cluster_val_labels, device, epochs_for_each_cluster,
                       load_weights=False):
    patience = 5
    model2_instances = []

    for cluster_id in range(K):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_dataset = Subset(train_dataset, cluster_indices)
        cluster_loader = DataLoader(cluster_dataset, batch_size=32, shuffle=True)

        model2_instance = Model2(model, 4, device).to(device)
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
        file_name = os.path.join(root_dir, 'src', 'features', 'mobilenetv2', 'model2',
                                 f'model2_cluster_{cluster_id}.pth')
        if os.path.exists(file_name) and load_weights:
            model2_instance.load_state_dict(torch.load(file_name))
            model2_instance.to(device)
            print(f"Model for cluster {cluster_id} loaded from {file_name}")
            model2_instances.append(model2_instance)
        else:
            model2_instance.load_state_dict(torch.load(file_name))
            model2_instance.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(list(model2_instance.new_layer.parameters()) +
                                         list(model2_instance.classifier.parameters()), lr=0.001)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

            best_val_loss = float('inf')
            count_early_stop = 0

            num_epochs = epochs_for_each_cluster
            for epoch in range(num_epochs):
                model2_instance.train()
                total_loss, correct_samples, total_samples = 0, 0, len(cluster_dataset)
                for images, labels in cluster_loader:
                    images, labels = images.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model2_instance(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    _, predicted = outputs.max(1)
                    correct_samples += (predicted == labels).sum().item()
                    total_loss += loss.item()

                train_accuracy = 100 * correct_samples / total_samples
                print(
                    f"Cluster {cluster_id}, Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(cluster_loader)}, Accuracy: {train_accuracy}%")

                model2_instance.eval()
                cluster_indices = [i for i, label in enumerate(cluster_val_labels) if label == cluster_id]
                cluster_val_dataset = Subset(val_dataset, cluster_indices)
                cluster_val_loader = DataLoader(cluster_val_dataset, batch_size=32, shuffle=True)
                val_loss, correct_samples, total_samples = 0, 0, len(cluster_val_dataset)
                with torch.no_grad():
                    for images, labels in cluster_val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model2_instance(images)
                        loss = criterion(outputs, labels)
                        _, predicted = outputs.max(1)
                        correct_samples += (predicted == labels).sum().item()
                        val_loss += loss.item()

                val_loss_avg = val_loss / len(cluster_val_loader)
                val_accuracy = 100 * correct_samples / total_samples
                print(f"Validation Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss_avg}, Accuracy: {val_accuracy}%")

                scheduler.step(val_loss_avg)

                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    count_early_stop = 0
                else:
                    count_early_stop += 1

                if count_early_stop >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            model2_instance.to(device)
            model2_instances.append(model2_instance)
            torch.save(model2_instance.state_dict(), file_name)
            print(f"Model for cluster {cluster_id} saved at {file_name}")

    return model2_instances
