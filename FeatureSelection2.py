# load train and test data
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

train_data_path = 'data/Train/'
trainX = pd.read_csv(train_data_path + 'X_train.txt', sep=" ", header=None).values
trainY = pd.read_csv(train_data_path + 'y_train.txt', header=None).values.ravel()

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
    testY = trainY[offset:]
    trainY = trainY[:offset]
    return trainX, trainY, testX, testY


trainX, trainY, testX, testY = seprate_test_train(trainX, trainY)
from sklearn.feature_selection import SelectKBest


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
        scores = cross_validate(self.clf, self.trainX, self.trainY, cv=cv, scoring=('neg_mean_squared_error'),
                                return_train_score=True)
        return -1 * np.max(scores['test_score'])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

class logistic_regression(classifier):
    def __init__(self):
        self.trainX, self.trainY, self.testX, self.testY = trainX, trainY, testX, testY
        self.train()

    def train(self, penalty='l1', C=np.inf, loss=True):
        self.clf = LogisticRegression(penalty=penalty, C=C, multi_class='multinomial', solver='saga')
        self.clf.fit(self.trainX, self.trainY)



def select_k_best(clf, k):
    fit = SelectKBest(k=k).fit(trainX, trainY)
    clf.trainX = fit.transform(trainX)
    clf.testX = fit.transform(testX)
    clf.train()
    return clf.MSE()
import matplotlib.pyplot as plt

feature_selection_types = ['SelectKBest', 'SelectPercentile', 'GenericUnivariateSelect']

def plotN(names, t, x):
    for i in range(len(names)):
        plt.plot(t, x[i])

    plt.legend(names, loc='upper left')

    plt.show()

def recursive_feature_elimination(classifier, k):
    selector = RFE(classifier.clf, k)
    selector = selector.fit(trainX, trainY)
    classifier.trainX = selector.transform(trainX)
    classifier.testX = selector.transform(testX)
    classifier.train()
    return classifier.MSE()


from sklearn.feature_selection import SelectFromModel


def l1_based(classifier, k):
    model = SelectFromModel(classifier.clf, max_features=k).fit(trainX, trainY)
    classifier.trainX = model.transform(trainX)
    classifier.testX = model.transform(testX)
    classifier.train()
    return classifier.MSE()


from sklearn.decomposition import PCA


def pca(classifier, k):
    model = PCA(n_components=k).fit(trainX)
    classifier.trainX = model.transform(trainX)
    classifier.testX = model.transform(testX)
    classifier.train()
    return classifier.MSE()

def feature_selection(classifier, ls):
    algorithms = [select_k_best, l1_based, pca]
    MSEs = list()
    for algorithm in algorithms:
        mses = list()
        for l in ls:
            classifier.__init__()
            mses.append(algorithm(classifier, l))
        MSEs.append(mses)
    return MSEs



def rf_feature_importance(classifier, k):
    features = classifier.feature_importace(k)[0]
    classifier.trainX = trainX[:, features]
    classifier.testX = testX[:, features]
    classifier.train()
    return classifier.MSE()



lr = logistic_regression()
ls = [5, 10, 50, 100, 561]
results = feature_selection(lr, ls)
plotN(['SelectKBest', 'select_form_model', 'pca'], ls, results)