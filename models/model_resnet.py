import torch
import torch.nn as nn
from torchvision.models import resnet50


class SerializableModule(nn.Module):
    def __init__(self):
        super(SerializableModule, self).__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


NUM_CLASSES = 10
class Model(SerializableModule):
    def __init__(self, weights_path=None):
        super(Model, self).__init__()

        model = resnet50()
        if weights_path is not None:
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict)

        num_features = model.fc.in_features
        model.fc = nn.Dropout(0.0)
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )
        self._model = model

    def forward(self, x):
        feat = self._model(x)
        logit = self.fc(feat)
        return feat, logit
