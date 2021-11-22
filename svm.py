# load train and test data
import pandas as pd

from sklearn.model_selection import cross_validate

train_data_path = 'data/Train/'
trainX = pd.read_csv(train_data_path + 'X_train.txt', sep=" ", header=None).values
trainY = pd.read_csv(train_data_path + 'y_train.txt', header=None).values

# test_data_path = 'data/Test/'
# testX = pd.read_csv(test_data_path + 'X_test.txt',sep=" ", header=None).values
# testY = pd.read_csv(test_data_path + 'y_test.txt', header=None).values

# if we don't have test data, run this cell
import numpy as np


def seprate_test_train(trainX, trainY, ratio=0.8, shuffle=False):
    if shuffle:
        train = np.zeros((trainX.shape[0], trainX.shape[1] + trainY.shape[1]))
        train[:, :-trainY.shape[1]] = trainX
        train[:, trainX.shape[1]:] = trainY
        np.random.shuffle(train)
        trainX = train[:, :-trainY.shape[1]]
        trainY = train[:, trainX.shape[1]:]
    offset = np.int(trainX.shape[0] * ratio)
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


from sklearn.svm import LinearSVC


class svm(classifier):
    def __init__(self):
        self.trainX, self.trainY, self.testX, self.testY = trainX, trainY, testX, testY
        self.train()

    def train(self, penalty='l2', C=np.inf, loss='squared_hinge'):
        dual = True
        if C != np.inf and C > 100:
            dual = False
        self.clf = LinearSVC(penalty=penalty, loss=loss, C=C, dual=dual)
        self.clf.fit(self.trainX, self.trainY.ravel())


def norm(clf, landas, ntype='l1'):
    Cs = [1 / i for i in landas]

    errors = list()
    zero_params = list()

    for C in Cs:
        clf.train(ntype, C)
        errors.append(clf.MSE())
        zero_params.append(clf.zero_parameter_number())
    return errors, zero_params


s = svm()
print('train accuracy is:')
print(s.train_accuracy())
print('test accuracy is:')
print(s.test_accuracy())

landas = [10 ** -4, 2 * 10 ** -4, 5 * 10 ** -4, 10 ** -3, 2 * 10 ** -3, 5 * 10 ** -3, 10 ** -2, 0.1, 1, 10, 100]
log_landa = np.log10(landas)

print('l1 norm: ')
errors, zero_params = norm(s, landas, 'l1')
plot('MSE for l1 ', log_landa, errors)
plot('Number of zero parameters for l1', log_landa, zero_params)
print('l2 norm: ')
errors, zero_params = norm(s, landas, 'l2')
plot('MSE for l2', log_landa, errors)
plot('Number of zero parameters for l2', log_landa, zero_params)
