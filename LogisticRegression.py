import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

train_data_path = 'data/Train/'
trainX = pd.read_csv(train_data_path + 'X_train.txt', sep=" ", header=None).values
trainY = pd.read_csv(train_data_path + 'y_train.txt', header=None).values


# if we don't have test data, run this cell


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


def make_standard():
    global trainX, testX
    sc = StandardScaler()
    trainX = sc.fit_transform(trainX)
    testX = sc.transform(testX)


make_standard()


class classifier:
    global trainX, trainY, testX, testY

    def __init__(self):
        self.trainX, self.trainY, self.testX, self.testY = trainX, trainY, testX, testY
        self.clf = None

    def train(self, penalty='l1', C=np.inf):
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


class logistic_regression(classifier):
    def __init__(self):
        self.trainX, self.trainY, self.testX, self.testY = trainX, trainY, testX, testY
        self.train()

    def train(self, penalty='l1', C=np.inf, solver='saga'):
        self.clf = LogisticRegression(penalty=penalty, C=C, multi_class='multinomial', solver=solver)
        self.clf.fit(self.trainX, self.trainY.ravel())


def plot(name, t, x):
    plt.plot(t, x)
    plt.legend([name], loc='upper left')
    plt.show()


def norm(clf, landas, ntype='l1'):
    Cs = [1 / i for i in landas]

    errors = list()
    zero_params = list()

    for C in Cs:
        clf.train(ntype, C)
        errors.append(clf.MSE())
        zero_params.append(clf.zero_parameter_number())
    return errors, zero_params


def change_data_size(Ks, clf):
    global trainX, trainY, testX, testY
    MSEs = list()
    for k in Ks:
        offset = np.int(trainX.shape[0] * k)
        clf.trainX = trainX[:offset, :]
        clf.trainY = trainY[:offset, :]
        clf.train()
        MSEs.append(clf.MSE())
    return MSEs


lr = logistic_regression()
print('train accuracy is:')
print(lr.train_accuracy())
print('test accuracy is:')
print(lr.test_accuracy())
Ks = [0.05, 0.1, 0.2, 0.5, 1]
errors = change_data_size(Ks, lr)
plot('Train size effect on MSE', Ks, errors)
