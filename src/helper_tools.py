import numpy as np
import librosa
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

def heatmap(X, y_labels=[], x_labels=[]):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(X, cmap=plt.cm.Blues, alpha=0.8)
    if len(x_labels)>0:
        ax.set_xticklabels(x_labels, minor=False)
        ax.set_xticks(np.arange(X.shape[1]) + 0.5, minor=False)
    if len(y_labels)>0:
        ax.set_yticklabels(y_labels, minor=False)
        ax.set_yticks(np.arange(X.shape[0]) + 0.5, minor=False)
    plt.show()

def mfcc_map(X):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(X, x_axis='time')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
