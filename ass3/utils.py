import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn


def read_data(filename):
    X = []
    Y = []
    with open(filename, "r") as index:
        for line in index:
            line = line.rstrip('\n')
            att = line.split(' ')
            att = [int(x) for x in att]
            x = att[:36]
            y = att[36]-1
            X.append(x)
            Y.append(y)
            # print(att)
            # print(x)
            # print(y)
            # print(f"Number of lines is {cnt}")

    X = np.array(X)
    Y = np.array(Y)
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X-mu)/std
    return X, Y


class dataset(Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.labels = Y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


def fit(model, train_loader, epochs=2, lr=0.1, criterion=nn.CrossEntropyLoss(), opt_func=torch.optim.SGD):
    optim = opt_func(model.parameters(), lr=lr)

    for epoch in range(0, epochs):

        print(f'Current epoch {epoch + 1}')
        epoch_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            inputs, targets = data
            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            print(f"Loss after data item {i} is {loss.item()}")

        print(f"The total loss after epoch {epoch} is {epoch_loss}")

    print("Training process has finished")


def test(model, test_loader):
    model.eval()
    total = 0
    for i, data in enumerate(test_loader, 0):

        inputs, targets = data
        outputs = model(inputs)
        y_pred_softmax = torch.log_softmax(outputs, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        correct_pred = (y_pred_tags == targets)
        total += correct_pred
        print(f"item {i} predicted {y_pred_tags}, actual {targets}")

    print(f"Accuracy is {total*100/len(test_loader)}")
    return total*100/len(test_loader)


class PCA:
    def __init__(self, X):
        cov_mat = np.cov(X, rowvar=False)
        d, u = np.linalg.eigh(cov_mat)
        self.U = np.asarray(u).T[::-1]
        self.D = d[::-1]

    def project(self, X):
        z = np.dot(X, np.asmatrix(self.U[:2]).T)
        return np.array(z)
