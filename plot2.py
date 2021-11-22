import matplotlib.pyplot as plt

feature_selection_types = ['SelectKBest', 'SelectPercentile', 'GenericUnivariateSelect']

def plotN(names, t, x):
    for i in range(len(names)):
        plt.plot(t, x[i])

    plt.legend(names, loc='upper left')

    plt.show()


plotN(['a','b'],[1,2,3,4,5],[[2,4,6,8,10], [1,4,9,16, 25]])