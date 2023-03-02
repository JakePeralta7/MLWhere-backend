import torch
import pandas as pd
from torch import nn
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score, accuracy_score, f1_score
import numpy as np


def eval_model(y_true, y_proba):
    y_preds = np.argmax(y_proba, axis=1)
    pos_probs = [prob[1] for prob in y_proba]
    acc = accuracy_score(y_true,  y_preds)
    fpr, tpr, _ = roc_curve(y_true, pos_probs)
    auc = roc_auc_score(y_true, pos_probs)
    f1 = f1_score(y_true, y_preds)
    roc_df = pd.DataFrame.from_dict(dict(fpr=fpr, tpr=tpr))
    print(acc, f1)


def score_the_model(test_loader, mdl, scoring=None):
    """
    Evaluate torch model performance based on a scoring method(s)
    """
    mdl.eval()
    output_dict = {}
    test_total = 0
    test_correct = 0
    with torch.no_grad():
        for test_samples, test_labels in test_loader:
            if torch.cuda.is_available():
                print("cuda??")
                test_samples = test_samples.cuda()
                test_labels = test_labels.cuda()
            test_outputs = mdl(test_samples)
            preds = torch.argmax(test_outputs, dim=1)
            test_total += test_labels.size(0)
            test_correct += (preds == test_labels).sum()
    if scoring is None or 'acc' in scoring:
        output_dict['acc'] = float(test_correct / test_total)
    # TODO: support more scoring methods
    return output_dict


def train_single_model(train_loader, model, test_loader=None, verbose=True):
    """
    Train the model, evaluate on the test set for every epoch (opt), return data for evaluation
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 5
    train_correct = 0
    train_total = 0
    train_accs = []
    test_accs = []
    loss_func = nn.NLLLoss()
    for epoch in range(num_epochs):
        for i, (batch_samples, batch_labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                if verbose:
                    print("cuda??")
                batch_samples = batch_samples.cuda()
                batch_labels = batch_labels.cuda()
            optimizer.zero_grad()
            outputs = model(batch_samples) # probability for malware/label = 1

            # outputs = sigma(outputs)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_func(outputs, batch_labels) #unsqueeze if bce
            loss.backward()
            optimizer.step()
            train_total += batch_labels.size(0)
            train_correct += (preds == batch_labels).sum()

            if verbose and i % 100 == 0:
                print('Epoch: [{}/{}], Loss: {:.4}'.format(epoch + 1, num_epochs, loss.item()))

        train_accs.append(float(train_correct) / train_total)
        train_correct = 0
        train_total = 0
        if test_loader:
            model.train()
            test_accs.append(score_the_model(test_loader, model, scoring=['acc'])['acc'])
    torch.save(model.state_dict(), 'model.pkl')
    if verbose:
        print("...Training Done...")
    return train_accs, test_accs