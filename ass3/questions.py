import csv
import utils
import model
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]

X_train, Y_train = utils.read_data("sat.trn")
X_train = X_train.astype(np.float32)

# Uncomment lines 15, 16, 24, 25 to apply PCA
# pca = utils.PCA(X_train)
# X_train = pca.project(X_train)

print(np.shape(X_train))
print(np.shape(Y_train))

X_test, Y_test = utils.read_data("sat.tst")
X_test = X_test.astype(np.float32)

# X_test = pca.project(X_test)
# X_test = X_test.astype(np.float32)

print(np.shape(X_test))
print(np.shape(Y_test))

train_data = utils.dataset(X=X_train, Y=Y_train)
train_loader = DataLoader(train_data)

test_data = utils.dataset(X=X_test, Y=Y_test)
test_loader = DataLoader(test_data)


# Finding accuracy of all 5 types of models from question 2
def find_accuracy(mdl):
    accuracies = []
    for lr in learning_rates:
        mdl.load_state_dict(torch.load('clean.pth'))                         # resets the model
        utils.fit(model=mdl, train_loader=train_loader, epochs=20, lr=lr)    # train
        accuracies.append(utils.test(model=mdl, test_loader=test_loader))    # test
    print(accuracies)
    return accuracies


performance = []
# Question 2 A
mlp_a = model.MLP_0(input_dim=X_train.shape[1])
torch.save(mlp_a.state_dict(), 'clean.pth')
performance.append(find_accuracy(mlp_a))

# Question 2 B
mlp_b = model.MLP_1(input_dim=X_train.shape[1], hidden_dim=2)
torch.save(mlp_b.state_dict(), 'clean.pth')
performance.append(find_accuracy(mlp_b))

# Question 2 C
mlp_c = model.MLP_1(input_dim=X_train.shape[1], hidden_dim=6)
torch.save(mlp_c.state_dict(), 'clean.pth')
performance.append(find_accuracy(mlp_c))

# Question 2 D
mlp_d = model.MLP_2(input_dim=X_train.shape[1], hidden_dim1=2, hidden_dim2=3)
torch.save(mlp_d.state_dict(), 'clean.pth')
performance.append(find_accuracy(mlp_d))

# Question 2 E
mlp_e = model.MLP_2(input_dim=X_train.shape[1], hidden_dim1=3, hidden_dim2=2)
torch.save(mlp_e.state_dict(), 'clean.pth')
performance.append(find_accuracy(mlp_e))


# saving the accuracy values in a csv file
with open("performance_analysis.csv", "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(performance)

# plotting accuracy vs learning rate for all models
t = [1, 2, 3, 4, 5]
my_xticks = [0.00001, 0.0001, 0.001, 0.01, 0.1]
plt.xticks(t, my_xticks)
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.plot(t, performance[0], label="HD = []")
plt.plot(t, performance[1], label="HD = [2]")
plt.plot(t, performance[2], label="HD = [6]")
plt.plot(t, performance[3], label="HD = [2, 3]")
plt.plot(t, performance[4], label="HD = [3, 2]")
plt.legend(loc="upper right")
plt.savefig('Accuracy_vs_LearningRate.png')

plt.figure().clear()

# plotting model vs accuracy for all learning rates
my_xticks = ["[]", "[2]", "[6]",
             "[2, 3]", "[3, 2]"]
plt.xticks(t, my_xticks)
performance_ = np.transpose(performance)
plt.xlabel("Hidden Dimensions")
plt.ylabel("Accuracy")
plt.plot(t, performance_[0], label="lr = 0.00001")
plt.plot(t, performance_[1], label="lr = 0.0001")
plt.plot(t, performance_[2], label="lr = 0.001")
plt.plot(t, performance_[3], label="lr = 0.01")
plt.plot(t, performance_[4], label="lr = 0.1")
plt.legend(loc="upper right")
plt.savefig('Accuracy_vs_model.png')

plt.figure().clear()

