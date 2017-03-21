import numpy as np
import pandas as pd
import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


def file_loader(path, duration=4, format='mp3', limit=None, csv_export=False):
    '''
    INPUT: Path (str), duration (in s)
    OUTPU: List of raw audio data (array), sampling rate (int)
    Takes in the path to audio files (mp3) and loads them as floating time series.
    '''
    songdirs = glob.glob(path+'*.'+format)
    raw_audio_data = []
    sr = None
    for song in songdirs[:limit]:
        X, sr = librosa.load(song, duration=duration)
        raw_audio_data.append(X)
    if csv_export:
        np.savetxt(path.split('/')[-2]+'.csv', np.array(raw_audio_data), delimiter=',')
    return raw_audio_data, sr

def mfcc_extractor(raw_audio_data, sample_rate=22050):
    '''
    Takes in raw audio data (time series) and loads the MFCC. It then calculates
    the mean for the respective cepstrals.
    '''
    mfcc_list = []
    for song in raw_audio_data:
        m_mfcc = np.mean(librosa.feature.mfcc(y=song, sr=sample_rate).T,axis=0)
        mfcc_list.append(m_mfcc)
    return np.array(mfcc_list)

def feature_extractor(path, duration=5, format='mp3', song_limit=None, artist_limit=None):
    m_mfccs = []
    labels = []
    artists = glob.glob(path+'*/')[:artist_limit]
    for i, subdir in enumerate(artists):
        raw_audio_data, sr = file_loader(subdir, duration=duration, format=format, limit=song_limit)
        mfcc_list = mfcc_extractor(raw_audio_data, sample_rate=sr)
        m_mfccs.append(mfcc_list)
        labels.append(np.ones(len(mfcc_list))*i)
    m_mfccs = reduce(lambda x, y: np.append(x, y, axis=0), m_mfccs)
    labels = reduce(lambda x, y: np.append(x, y, axis=0), labels)
    return m_mfccs, labels


def multi_cv(X, y):
    '''
    Takes in a 2d array with data from multiple labels and a model and runs
    a K-fold cross validation on it.
    '''
    # # Determine how many samples per label are present
    # sf = StratifiedKFold(n_splits=5)
    # for split in sf.split(X, y):
    result = []
    sf = StratifiedKFold(n_splits=5, shuffle=True)
    for train, test in sf.split(X, y):
        classifier = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        classifier.fit(X[train], y[train])
        result.append(classifier.score(X[test], y[test]))
    return result

def shuffler(X, y):
    shuffler = np.array(range(len(X)))
    np.random.shuffle(shuffler)
    X_shuffled = X[shuffler]
    y_shuffled = y[shuffler]
    return X_shuffled, y_shuffled


if __name__ == '__main__':
    X, y = feature_extractor('../data/', song_limit=None, artist_limit=10)

    # X, y = shuffler(X,y)

    result = multi_cv(X, y)
    print np.mean(result)


    # svc_classifier = SVC()
    # print "SVC score: ", cross_val_score(svc_classifier, X, y, cv=5).mean()

    # logr_classifier = LogisticRegression()
    # print cross_val_score(logr_classifier, X, y, cv=5).mean()

    # rndmf_classifier = RandomForestClassifier()
    # print "Random forest score: ", cross_val_score(rndmf_classifier, X, y, cv=5).mean()
