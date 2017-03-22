import numpy as np
from helper_tools import shuffler, heatmap, csv_exporter
import pandas as pd
import glob
import os
import librosa
import librosa.display
from time import time
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
    start = time()
    songdirs = glob.glob(path+'*.'+format)
    raw_audio_data = []
    sr = None
    for song in songdirs[:limit]:
        X, sr = librosa.load(song, duration=duration)
        raw_audio_data.append(X)
    if csv_export:
        csv_exporter(raw_audio_data, path, songdirs, sr)
    print 'File loader for one artist done', time()-start
    return raw_audio_data, sr, songdirs

def mfcc_extractor(raw_audio_data, sample_rate=22050):
    '''
    Takes in raw audio data (time series) and loads the MFCC. It then calculates
    the mean for the respective cepstrals.
    '''
    start = time()
    mfcc_list = []
    for song in raw_audio_data:
        m_mfcc = np.mean(librosa.feature.mfcc(y=song, sr=sample_rate).T,axis=0)
        mfcc_list.append(m_mfcc)
    print 'MFCC for one artist done', time()-start
    return np.array(mfcc_list)

def feature_extractor(path, duration=5, format='mp3', song_limit=None, artist_limit=None):
    m_mfccs = []
    labels = []
    song_ids = []
    artists = glob.glob(path+'*/')[:artist_limit]
    for i, subdir in enumerate(artists):
        raw_audio_data, sr, songdirs = file_loader(subdir, duration=duration, format=format, limit=song_limit)
        mfcc_list = mfcc_extractor(raw_audio_data, sample_rate=sr)
        m_mfccs.append(mfcc_list)
        labels.append(np.ones(len(mfcc_list))*i)
        song_ids.append(songdirs[:song_limit])
    m_mfccs = reduce(lambda x, y: np.append(x, y, axis=0), m_mfccs)
    labels = reduce(lambda x, y: np.append(x, y, axis=0), labels)
    song_ids = reduce(lambda x, y: np.append(x, y, axis=0), song_ids)
    return m_mfccs, labels, song_ids

def csv_feature_extractor(path, duration=5, song_limit=None):
    '''
    Takes in a directory with csv files created by the file_loader and extracts the features.
    '''
    m_mfccs = []
    labels = []
    song_ids = []
    raw_artistfiles = glob.glob(path+'/raw_data/'+'*.npy')
    meta_artistfiles = glob.glob(path+'/meta_info/'+'*.csv')
    for i, raw_file in enumerate(raw_artistfiles):
        start = time()
        raw_audio_data = np.load(raw_file)
        mfcc_list = mfcc_extractor(raw_audio_data[:song_limit])
        m_mfccs.append(mfcc_list)
        labels.append(np.ones(len(mfcc_list))*i)
        print 'Convertion of audio done for one artist', time()-start
    for meta_file in meta_artistfiles:
        songdirs = np.loadtxt(meta_file, dtype=str)
        song_ids.append(songdirs[:song_limit])
    m_mfccs = reduce(lambda x, y: np.append(x, y, axis=0), m_mfccs)
    labels = reduce(lambda x, y: np.append(x, y, axis=0), labels)
    song_ids = reduce(lambda x, y: np.append(x, y, axis=0), song_ids)
    return m_mfccs, labels, song_ids


def one_artist_feature_extractor(artist_dir, duration=5, format='mp3', song_limit=None, artist_limit=None):
    raw_audio_data, sr, songdirs = file_loader(artist_dir, duration=duration, format=format, limit=song_limit)
    m_mfccs = mfcc_extractor(raw_audio_data, sample_rate=sr)
    song_ids = songdirs[:song_limit]
    return m_mfccs, song_ids

def multi_cv(X, y):
    '''
    Takes in a 2d array with data from multiple labels and a model and runs
    a K-fold cross validation on it.
    '''
    result = []
    sf = StratifiedKFold(n_splits=5, shuffle=True)
    for train, test in sf.split(X, y):
        classifier = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        classifier.fit(X[train], y[train])
        result.append(classifier.score(X[test], y[test]))
    return result

def prediction_analyser(model, X, y, songids):
    '''
    INPUT: model, 2d array, 1d array, list of lists
    OUTPUT: song_ids with correct predictions and false predictions
    '''
    X_train, X_test, y_train, y_test, song_train, song_test = train_test_split(X, y, songids)
    classifier = model()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    correct = song_test[predictions==y_test]
    wrong = song_test[predictions!=y_test]
    X_correct = X_test[predictions==y_test]
    X_wrong = X_test[predictions!=y_test]
    return correct, wrong, X_correct, X_wrong


if __name__ == '__main__':
    # X, y, song_ids = feature_extractor('../data/', song_limit=50, artist_limit=2)
    # X, y = shuffler(X,y)

    X, y, song_ids = csv_feature_extractor('./', song_limit=10)
    X, y = shuffler(X,y)


    # correct, wrong, X_correct, X_wrong = prediction_analyser(RandomForestClassifier, X, y, song_ids)
    # heatmap(X_wrong, y_labels=wrong)


    # result = multi_cv(X, y)
    # print np.mean(result)


    rndmf_classifier = RandomForestClassifier()
    print "Random forest score: ", cross_val_score(rndmf_classifier, X, y, cv=5).mean()


    # svc_classifier = SVC()
    # print "SVC score: ", cross_val_score(svc_classifier, X, y, cv=5).mean()

    # logr_classifier = LogisticRegression()
    # print cross_val_score(logr_classifier, X, y, cv=5).mean()
