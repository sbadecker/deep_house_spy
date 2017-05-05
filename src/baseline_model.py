import numpy as np
import pandas as pd
import glob
import os
import librosa
from helper_tools import shuffler, heatmap, csv_exporter
from time import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing
from functools import partial

#########################################
############# File loader ###############
#########################################

def file_loader(path, format='mp3', duration=5, offset=0.0, song_limit=None, csv_export=False):
    '''
    INPUT: Path (str), duration (in s)
    OUTPUT: List of raw audio data (array), sampling rate (int)

    Takes in the path to audio files (mp3) and loads them as floating time series.
    When csv_export is enabled it calls the csv_exporter which stores the
    audio and meta data as pickle and csv files.
    '''
    start = time()
    songdirs = glob.glob(path+'*.'+format)
    raw_audio_data = []
    sr = None
    for song in songdirs[:song_limit]:
        X, sr = librosa.load(song, duration=duration, offset=offset)
        raw_audio_data.append(X)
    if csv_export:
        csv_exporter(raw_audio_data, path, songdirs)
    print 'File loader for one artist done', time()-start
    return raw_audio_data, sr, songdirs

def parallel_file_loader(path, format='mp3', duration=None, offset=0.0, song_limit=None, sample_rate=22050, csv_export=True, pool_size=4):
    '''
    Takes in the path to audio files (mp3) and loads them as floating time series.
    When csv_export is enabled it calls the csv_exporter which stores the
    audio and meta data as pickle and csv files.

    Can run use multiple cores (specified by pool_size).
    '''
    start = time()
    raw_audio_data = []
    songdirs = glob.glob(path+'*.'+format)[:song_limit]
    pool = multiprocessing.Pool(processes=pool_size)
    X = pool.map(partial(librosa.load, duration=duration, offset=offset, sr=sample_rate), songdirs)
    pool.close()
    pool.join()
    sr = X[0][1]
    print 'Audio transformation done', time()-start
    for song in X:
        raw_audio_data.append(song[0])
    if csv_export:
        csv_exporter(raw_audio_data, path, songdirs)
    return raw_audio_data, sr, songdirs


#########################################
########### Feature extractor ###########
#########################################

def feature_extractor(raw_audio_data, n_mfcc=20, sample_rate=22050, full_mfccs=False):
    '''
    Takes in raw audio data (time series) and loads the MFCC. It then calculates
    the mean for the respective cepstrals.
    '''
    start = time()
    feature_list = []
    for song in raw_audio_data:
        if full_mfccs:
            mfcc = librosa.feature.mfcc(y=song, sr=sample_rate, n_mfcc=n_mfcc)
        else:
            mfcc = np.mean(librosa.feature.mfcc(y=song, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
        feature_list.append(mfcc)
    return np.array(feature_list)


#########################################
####### File processing scripts  ########
#########################################

def batch_extractor(path, duration=5, format='mp3', song_limit=None, artist_limit=None, n_mfcc=20):
    feature_list = []
    labels = []
    song_ids = []
    artists = sorted(glob.glob(path+'*/'))[:artist_limit]
    for i, subdir in enumerate(artists):
        raw_audio_data, sr, songdirs = file_loader(subdir, duration=duration, format=format, song_limit=song_limit)
        features = feature_extractor(raw_audio_data[:song_limit], n_mfcc=n_mfcc)
        feature_list.append(features)
        labels.append(np.ones(len(features))*i)
        song_ids.append(songdirs[:song_limit])
    feature_list = reduce(lambda x, y: np.append(x, y, axis=0), feature_list)
    labels = reduce(lambda x, y: np.append(x, y, axis=0), labels)
    song_ids = reduce(lambda x, y: np.append(x, y, axis=0), song_ids)
    return feature_list, labels, song_ids

def csv_batch_extractor(path, duration=5, song_limit=None, artist_limit=None, n_mfcc=20, full_mfccs=False):
    '''
    Takes in a directory with csv files created by the file_loader and extracts the features.
    '''
    feature_list = []
    labels = []
    song_ids = []
    raw_artistfiles = sorted(glob.glob(path+'raw_data/'+'*.npy'))[:artist_limit]
    meta_artistfiles = sorted(glob.glob(path+'meta_info/'+'*.csv'))[:artist_limit]
    for i, raw_file in enumerate(raw_artistfiles):
        start = time()
        raw_audio_data = np.load(raw_file)
        features = feature_extractor(raw_audio_data[:song_limit], n_mfcc=n_mfcc, full_mfccs=full_mfccs)
        feature_list.append(features)
        labels.append(np.ones(len(features))*i)
    for meta_file in meta_artistfiles[:artist_limit]:
        songdirs = np.loadtxt(meta_file, dtype=str)
        song_ids.append(songdirs[:song_limit])
    feature_list = reduce(lambda x, y: np.append(x, y, axis=0), feature_list)
    labels = reduce(lambda x, y: np.append(x, y, axis=0), labels)
    song_ids = reduce(lambda x, y: np.append(x, y, axis=0), song_ids)
    return feature_list, labels, song_ids

def one_artist_feature_extractor(artist_dir, duration=5, format='mp3', song_limit=None, artist_limit=None):
    raw_audio_data, sr, songdirs = file_loader(artist_dir, duration=duration, format=format, limit=song_limit)
    m_mfccs = mfcc_extractor(raw_audio_data, sample_rate=sr)
    song_ids = songdirs[:song_limit]
    return m_mfccs, song_ids


#########################################
############ Analysis tools #############
#########################################

def multi_cv(X, y, model= OneVsRestClassifier(RandomForestClassifier(random_state=0))):
    '''
    Takes in a 2d array with data from multiple labels and a model and runs
    a K-fold cross validation on it.
    '''
    result = []
    sf = StratifiedKFold(n_splits=5, shuffle=True)
    for train, test in sf.split(X, y):
        model.fit(X[train], y[train])
        result.append(model.score(X[test], y[test]))
    return result

def prediction_analyser(model, X, y, songids):
    '''
    INPUT: model, 2d arr, 1d arr, list of lists
    OUTPUT: arr, arr, arr, arr

    Splits the input data into train, and test, trains the given model and then
    runs the test set on it. It then splits the song id and the test data into
    cases that the model got wrong and cases that the model got right.
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


#########################################
############ Artist loader ##############
#########################################

def batch_artist_loader(path):
    '''
    INPUT: path (str)
    OUTPUT: Pickle (raw audio data), csv files (songdirs)

    Loads the data for all artistfolders in the path into raw audio data arrays
    and exports them as pickle files.
    '''
    for artist in glob.glob(path+'*/'):
        file_loader(artist, duration=None, song_limit=None, csv_export=True)


if __name__ == '__main__':
    pass
