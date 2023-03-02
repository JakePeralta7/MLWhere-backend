import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split as tt_split


class ClampDataset(Dataset):
    def __init__(self, features, labels,  transform=None):
        self.labels = labels
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vec = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feature_vec = self.transform(feature_vec).float()
            label = self.transform(label).long()
        return feature_vec, label


class ClampTestDataset(Dataset):
    def __init__(self, features,  transform=None):
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vec = self.features[idx]
        if self.transform:
            feature_vec = self.transform(feature_vec).float()
        return feature_vec


class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        table, label = self.subset[idx]
        if self.transform:
            table = self.transform(table)
        return table, label

    def __len__(self):
        return len(self.subset)


def create_dataloader(features, labels, transform, bs, shuffle):
    if labels is not None:
        cds = ClampDataset(features=features, labels=labels, transform=transform)
    else:
        cds = ClampTestDataset(features=features, transform=transform)
    return DataLoader(cds, batch_size=bs, shuffle=shuffle, num_workers=1, pin_memory=True), cds


def preprocess_data(csv_path):
    """
    Normalize data and split it to train/test pairs.
    """
    df = pd.read_csv(csv_path)
    labels = df['class'].values
    x = df.drop(columns=['packer_type', 'class']).values
    scaler = StandardScaler()
    features = scaler.fit_transform(x)
    X_train, X_test, y_train, y_test = tt_split(features, labels, random_state=42)
    return X_train, X_test, y_train, y_test


def train_test_split(torch_dataset, batch_size, shuffle=False, train_ratio=0.7):
    trainset, valset = random_split(torch_dataset,
                                    [int(len(torch_dataset) * train_ratio),
                                     len(torch_dataset) - int(len(torch_dataset) * train_ratio)],
                                    generator=torch.Generator().manual_seed(42))
    trainset, valset = SubsetDataset(trainset, None), SubsetDataset(valset, None)
    return DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True), trainset, \
           DataLoader(valset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True), valset
