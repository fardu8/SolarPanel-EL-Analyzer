import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights


class Model1:
    def __init__(self, device, load_weights=False):
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 4)
        self.load_weights = load_weights
        self.device = device


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class Model2(nn.Module):
    def __init__(self, base_model, num_classes, device):
        super(Model2, self).__init__()
        self.features = base_model.features
        self.new_layer = nn.Linear(1280, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.device = device

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.new_layer(x)
        x = self.classifier(x)
        return x


class SolarCellDefectDetector:
    def __init__(self, model2_instances, model1, device, kmeans):
        self.model2_instances = model2_instances
        self.model1 = model1
        self.device = device
        self.kmeans = kmeans

    def detect_defect(self, image):
        self.model1.eval()
        feature_extractor = FeatureExtractor(self.model1).to(self.device)
        image = image.to(self.device)
        features = feature_extractor(image).detach().cpu().numpy()

        cluster_id = self.kmeans.predict(features.reshape(1, -1))[0]

        model2 = self.model2_instances[cluster_id].to(self.device)
        model2.eval()
        output = model2(image)
        _, predicted_class = torch.max(output, 1)

        return predicted_class.item()
