import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier


def create_svm_model():

    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', OneVsRestClassifier(SVC(kernel='rbf', probability=True, C=10, gamma=0.01)))
    ])

    return svm_pipeline


def extract_features_for_svm(model, dataloader, device):

    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)

            feature = model.conv_layers(data)
            feature = feature.view(feature.size(0), -1)

            features.extend(feature.cpu().numpy())
            labels.extend(target.numpy())

    return np.array(features), np.array(labels)