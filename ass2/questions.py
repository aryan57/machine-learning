import pandas as pd
import matplotlib.pyplot as plt
import utils
import knn_model

# Reading data
dataset = utils.read_data('spam_ham_dataset.csv')

# splitting train and test set
x_train, x_test = utils.split_data(dataset)

# initializing knn classifier
model = knn_model.KnnClassifier(x_train, x_test)

# using euclidean distance for finding nearest neighbours
result = model.accuracy(utils.euclidean_dist)
print(result)

best_result = 0
best_k = 0
for i in range(len(result)):
    if result[i] > best_result:
        best_k = i
        best_result = result[i]

print("Best accuracy of "+str(best_result)+" occurs at k = "+str(best_k))

x = [i for i in range(len(result))]

x = x[1:]
result = result[1:]

plt.title("Accuracy vs K(euclidean)")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.plot(x, result, color="red")
plt.savefig('Accuracy_vs_K_euclidean.png')

plt.figure().clear()
##################################################################

# using manhattan distance for finding nearest neighbours
result = model.accuracy(utils.manhattan_dist)
print(result)

best_result = 0
best_k = 0
for i in range(len(result)):
    if result[i] > best_result:
        best_k = i
        best_result = result[i]

print("Best accuracy of "+str(best_result)+" occurs at k = "+str(best_k))

x = [i for i in range(len(result))]

x = x[1:]
result = result[1:]

print(x[0])
print(result[0])

plt.title("Accuracy vs K(Manhattan)")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.plot(x, result, color="red")
plt.savefig('Accuracy_vs_K_manhattan.png')

plt.figure().clear()
##################################################################

# using cosine similarity for finding nearest neighbours
result = model.accuracy(utils.cosine_similarity)
print(result)

best_result = 0
best_k = 0
for i in range(len(result)):
    if result[i] > best_result:
        best_k = i
        best_result = result[i]

print("Best accuracy of "+str(best_result)+" occurs at k = "+str(best_k))

x = [i for i in range(len(result))]

x = x[1:]
result = result[1:]

print(x[0])
print(result[0])

plt.title("Accuracy vs K(cosine)")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.plot(x, result, color="red")
plt.savefig('Accuracy_vs_K_cosine.png')

plt.figure().clear()