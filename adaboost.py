# load train and test data
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
train_data_path = 'data/Train/'
trainX = pd.read_csv(train_data_path + 'X_train.txt',sep=" ", header=None).values
trainY = pd.read_csv(train_data_path + 'y_train.txt', header=None).values

# test_data_path = 'data/Test/'
# testX = pd.read_csv(test_data_path + 'X_test.txt',sep=" ", header=None).values
# testY = pd.read_csv(test_data_path + 'y_test.txt', header=None).values

# if we don't have test data, run this cell
import numpy as np
def seprate_test_train(trainX, trainY, ratio=0.8, shuffle=False):
    if shuffle:
        train = np.zeros((trainX.shape[0], trainX.shape[1]+trainY.shape[1]))
        train[:,:-trainY.shape[1]] = trainX
        train[:,trainX.shape[1]:] = trainY
        np.random.shuffle(train)
        trainX = train[:,:-trainY.shape[1]]
        trainY = train[:,trainX.shape[1]:]
    offset= np.int(trainX.shape[0]*ratio)
    testX = trainX[offset:, :]
    trainX = trainX[:offset, :]
    testY = trainY[offset:, :]
    trainY = trainY[:offset, :]
    return trainX, trainY, testX, testY

trainX, trainY, testX, testY = seprate_test_train(trainX, trainY)

from sklearn.preprocessing import StandardScaler
def make_standard():
    global trainX, testX
    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.transform(testX)
make_standard()

import matplotlib.pyplot as plt

def plot(name, t, x):
    plt.plot(t, x)
    plt.legend([name], loc='upper left')
    plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


class classifier:
    global trainX, trainY, testX, testY

    def __init__(self):
        self.trainX, self.trainY, self.testX, self.testY = trainX, trainY, testX, testY
        self.clf = None

    def train(self):
        pass

    def train_accuracy(self):
        return self.clf.score(self.trainX, self.trainY)

    def test_accuracy(self):
        return self.clf.score(self.testX, self.testY)

    def MSE(self):
        predictY = self.clf.predict(self.testX)
        return mean_squared_error(self.testY, predictY)

    def zero_parameter_number(self):
        return np.where(self.clf.coef_ < 10 ** -6)[0].shape[0]

    def cross_validation(self, cv=5):
        scores = cross_validate(self.clf, self.trainX, self.trainY.ravel(), cv=cv, scoring=('neg_mean_squared_error'),
                                return_train_score=True)
        return -1 * np.max(scores['test_score'])

from sklearn.ensemble import AdaBoostClassifier


class adaboost(classifier):
    def __init__(self):
        self.trainX, self.trainY, self.testX, self.testY = trainX, trainY, testX, testY
        self.train()

    def train(self, n_estimators=20, learning_rate=0.1):
        self.clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        self.clf.fit(self.trainX, self.trainY.ravel())


ad = adaboost()
print('train accuracy is:')
print(ad.train_accuracy())
print('test accuracy is:')
print(ad.test_accuracy())

# cross validation for n_estimators
n_estimators = [1, 5, 10, 20, 50, 100]
MSEs = list()
for n_estimator in n_estimators:
    ad.train(n_estimators=n_estimator)
    MSEs.append(ad.cross_validation())
plot('Number of trees affect on MSE: ', n_estimators, MSEs)

# cross validation for learning_rate
# rates = [0.01, 0.1, 1, 10]
# rates_log = np.log10(rates)
# MSEs = list()
# for rate in rates:
#     ad.train(learning_rate=rate)
#     MSEs.append(ad.cross_validation())
# plot('Number of trees affect on MSE: ', rates_log, MSEs)