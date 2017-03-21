import numpy as np
import pandas as pd
import glob
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


def file_loader(path, duration=1, format='mp3', limit=10, csv_export=False):
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

def feature_extractor(path, duration=5, format='mp3', limit=20):
    m_mfccs = []
    for subdir in glob.glob(path+'*/'):
        raw_audio_data, sr = file_loader(subdir, duration=duration, format=format, limit=limit)
        mfcc_list = mfcc_extractor(raw_audio_data, sample_rate=sr)
        m_mfccs.append(mfcc_list)
    m_mfccs = reduce(lambda x, y: np.append(x, y, axis=0), m_mfccs)
    return m_mfccs


if __name__ == '__main__':
    X = feature_extractor('../data/')
    y = np.append(np.zeros(100), [np.ones(100), np.ones(100)*2])
    # y = np.append(np.zeros(20), np.ones(20))

    shuffler = np.array(range(len(X)))
    np.random.shuffle(shuffler)
    X = X[shuffler]
    y = y[shuffler]

    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # svc_classifier = SVC()
    # print "SVC score: ", cross_val_score(svc_classifier, X, y, cv=5).mean()

    # logr_classifier = LogisticRegression()
    # print cross_val_score(logr_classifier, X, y, cv=5).mean()

    multi = OneVsRestClassifier(RandomForestClassifier(random_state=0))
    # rndmf_classifier = RandomForestClassifier()
    print "Random forest score: ", cross_val_score(multi, X, y, cv=5).mean()
