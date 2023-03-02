import torch
import data_extract as de
from torch import nn
import numpy as np
from train_eval import train_single_model, score_the_model
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import joblib

ACTIVATIONS = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), 'logsoftmax': nn.LogSoftmax(dim=1),
               'identity': nn.Identity(), 'softmax': nn.Softmax(dim=1)}


class MLP(nn.Module):
    """
    A flexible MLP Class
    """

    def __init__(self, in_dim,
                 mlp_hidden_dims, output_dim,
                 activation_type, final_activation_type, dropout=0, bs=64):
        super(MLP, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_dim = in_dim
        self.bs = bs
        activation = ACTIVATIONS[activation_type]
        final_activation = ACTIVATIONS[final_activation_type]
        mlp_layers = []
        if (len(mlp_hidden_dims)) > 0:
            mlp_layers.extend([nn.Linear(in_dim, mlp_hidden_dims[0]), activation])
            for i in range(len(mlp_hidden_dims) - 1):
                mlp_layers.extend([nn.Linear(mlp_hidden_dims[i], mlp_hidden_dims[i + 1]), activation])
                if dropout:
                    mlp_layers.append(nn.Dropout(dropout))
            mlp_layers.extend([nn.Linear(mlp_hidden_dims[-1], output_dim), final_activation])
        else:
            mlp_layers.extend([nn.Linear(in_dim, output_dim), final_activation])
        self.mlp = nn.Sequential(*mlp_layers)
        print(self.mlp)

    def forward(self, x):
        out = self.mlp(x)
        return out

    def fit(self, x_train, y_train, dloader=None, reset_weights=True):
        """
        Fits the model to the dataset
        """
        if reset_weights:
            for layer in self.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        if not dloader:
            dloader, trainset = de.create_dataloader(x_train, y_train, torch.tensor, self.bs, False)
        train_single_model(dloader, self, verbose=False)

    def predict(self, x_test, y_test=None, test_loader=None, proba=False, scoring=None):
        """
        Predicts labels/probas to x_test. if y_test is not none also scores.
        """
        # y_test passed as a parameter- scoring
        if test_loader is None and y_test is not None:
            test_loader, testset = de.create_dataloader(x_test, y_test, torch.tensor, self.bs, False)
        elif not test_loader:
            test_loader, testset = de.create_dataloader(features=x_test, labels=None, transform=torch.tensor,
                                                        bs=self.bs, shuffle=False)
        if y_test is not None:  # scoring
            score_dict = score_the_model(test_loader, self, scoring=scoring)
            return score_dict['acc']  # TODO: support other scoring rules.
        else:  # predicting
            self.eval()
            all_preds = []
            with torch.no_grad():
                for test_samples in test_loader:
                    if torch.cuda.is_available():
                        print("cuda??")
                        test_samples = test_samples.cuda()
                    test_outputs = self(test_samples)
                    if proba:
                        preds = torch.exp(test_outputs)
                    else:
                        preds = torch.argmax(test_outputs, dim=1)
                    all_preds.extend(preds.tolist())
            return all_preds

    def predict_proba(self, x_test):
        return self.predict(x_test, proba=True)

    def score(self, x_test, y_test):
        return self.predict(x_test, y_test)


class StackingEnsemble:
    """
    Stacking Ensemble model
    """

    def __init__(self, models, folds, final_model, reduction_level):
        super(StackingEnsemble, self).__init__()
        self.base_models = models  # Collection of already initiated models
        self.final_model = final_model
        self.pca_components = reduction_level
        self.folds = folds
        self.pca = None

    def fit(self, X_train, y_train):
        kfolder = KFold(n_splits=self.folds, shuffle=True, random_state=42)
        splits = kfolder.split(X_train, y_train)
        x_formeta = X_train.copy()
        if self.pca_components != 0:
            self.pca = PCA(n_components=self.pca_components)
            x_formeta = self.pca.fit_transform(X_train)

        XM_train = np.hstack((x_formeta.copy(), np.full((X_train.shape[0], len(self.base_models)), -1)))

        for fold_train_indices, fold_test_indices in splits:
            for i, model in enumerate(self.base_models):
                x_fold_train = X_train[fold_train_indices]
                x_fold_test = X_train[fold_test_indices]
                y_fold_train = y_train[fold_train_indices]
                model.fit(x_fold_train, y_fold_train)
                probs = model.predict_proba(x_fold_test)
                pos_probs = [prob[1] for prob in probs]
                XM_train[fold_test_indices, (x_formeta.shape[1] + i)] = np.array(pos_probs).reshape(-1, len(pos_probs))
        self.final_model.fit(XM_train, y_train)

    def predict(self, X_test, proba=False):
        x_test_formeta = X_test.copy()
        if self.pca_components != 0:
            n_pcs = self.pca.components_.shape[0]
            most_important = [np.abs(self.pca.components_[i]).argmax() for i in range(n_pcs)]
            x_test_formeta = X_test[:, most_important]
        XM_test = np.hstack((x_test_formeta.copy(), np.full((X_test.shape[0], len(self.base_models)), -1)))
        for i, model in enumerate(self.base_models):
            probs = model.predict_proba(X_test)
            pos_probs = [prob[1] for prob in probs]
            XM_test[:, (x_test_formeta.shape[1] + i)] = np.array(pos_probs).reshape(-1, len(pos_probs))
        final_preds = self.final_model.predict_proba(XM_test)
        if not proba:
            final_preds = np.argmax(final_preds, axis=1)
        return final_preds

    def predict_proba(self, X_test):
        return self.predict(X_test, proba=True)

    def save(self):
        joblib.dump(self, 'stacking_model.pkl')
