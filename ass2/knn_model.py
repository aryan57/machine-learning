import utils
import numpy as np
eps = 0.000000001


# The knn class
class KnnClassifier:
    def __init__(self, x_train=None, x_test=None):
        self.x_train = x_train
        self.x_test = x_test

    # simple classifier without weights
    def classify(self, item, dist_func):

        # finds all the neighbours and their distances
        neighbours = np.zeros(shape=(len(self.x_train), 2))
        cnt = 0
        for y in self.x_train:
            neighbours[cnt] = [dist_func(item['text'], y['text']), y['label']]
            cnt += 1

        # sort all the neighbours by distance
        neighbours = neighbours[neighbours[:, 0].argsort()]

        # this vector maintains the frequency of classes for all k
        count = np.zeros([neighbours.shape[0]+1, 2], dtype=int)

        for i in range(neighbours.shape[0]):
            count[i + 1] = count[i]
            count[i+1][int(neighbours[i][1])] += 1

        # assigning the class based on the higher frequency class
        result = []
        for i in count:
            if i[0] >= i[1]:
                result.append(0)
            else:
                result.append(1)
        return result

    # weights introduced as inverse of distance
    def classify_weighted(self, item, dist_func):
        neighbours = np.zeros(shape=(len(self.x_train), 2))
        cnt = 0

        # finds all the neighbours and their distances
        for y in self.x_train:
            neighbours[cnt] = [dist_func(item['text'], y['text'])+eps, y['label']]
            cnt += 1

        neighbours = neighbours[neighbours[:, 0].argsort()]

        numerator, denominator = 0, 0

        aggregate = np.zeros(neighbours.shape[0]+1)

        # weighted method numerator is sum of weight into class
        # denominator is sum of weights
        for i in range(neighbours.shape[0]):
            denominator += 1/(neighbours[i][0]*neighbours[i][0])
            numerator += (neighbours[i][1])/(neighbours[i][0]*neighbours[i][0])
            aggregate[i + 1] = numerator/denominator

        # results based on the weighted mean
        result = []
        for i in aggregate:
            if i > 0.5:
                result.append(1)
            else:
                result.append(0)

        return result

    def accuracy(self, dist_func):
        correct = np.zeros(len(self.x_train)+1)
        total = len(self.x_test)
        counter = 0

        # call the classifier for all test samples
        for item in self.x_test:
            print("processing item "+str(counter))
            result = self.classify_weighted(item, dist_func)

            result = (result == item['label'])

            print(result)
            counter += 1
            correct += result
        return correct*100/total
