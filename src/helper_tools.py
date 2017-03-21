import numpy as np
import matplotlib.pyplot as plt

def shuffler(X, y):
    shuffler = np.array(range(len(X)))
    np.random.shuffle(shuffler)
    X_shuffled = X[shuffler]
    y_shuffled = y[shuffler]
    return X_shuffled, y_shuffled

def plotter(X):
    fig = plt.figure()
    n_plots = len(X)
    for i in range(n_plots):
        plotcode = n_plots*100+10+i+1
        ax = fig.add_subplot(plotcode)
        ax.plot(X[i])
    plt.show()

def heatmap(X):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(X, cmap=plt.cm.Blues, alpha=0.8)
    splt.show()
