import numpy as np
import glob
import os
import librosa
from time import time
import multiprocessing
from functools import partial
from baseline_model import csv_batch_extractor
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


#########################################
############# Main engine ###############
#########################################

def main_engine(path, splits=1, song_limit=None, artist_limit=None):
    '''
    INPUT: Directory of Artist directories with pickles in them
    OUTPUT: X, y, song_ids
    Loads in all songs from artists in the specified path, and handles the
    splitting, feature extraction and snippet selection process.
    '''
    X = []
    y = []
    song_ids = []
    raw_artistfiles = sorted(glob.glob(path+'raw_data/'+'*.npy'))[:artist_limit]
    meta_artistfiles = sorted(glob.glob(path+'meta_info/'+'*.csv'))[:artist_limit]
    for i, raw_artist in enumerate(raw_artistfiles):
        raw_audio_data = np.load(raw_artist)[:song_limit]
        songdirs = np.loadtxt(meta_artistfiles[i], dtype=str)[:song_limit]
        for j, song in enumerate(raw_audio_data):
            X_song = []
            y_song = []
            song_ids_song = []
            snippets = np.split(song[song.shape[0]%splits:], splits)
            for snippet in snippets:
                snippet_features_raw = snippet_feature_extractor(snippet)
                snippet_features = snippet_selector(snippet_features_raw)
                X_song.append(snippet_features)
                y_song.append(i)
                song_ids_song.append(songdirs[j])
            X.append(X_song)
            y.append(y_song)
            song_ids.append(song_ids_song)
            if j != 0 and (j+1) % 10 == 0:
                print '%d songs done' %(j+1)
    return np.array(X), np.array(y), np.array(song_ids)


#########################################
########### Snippet handlers ############
#########################################

def snippet_feature_extractor(snippet, n_mfcc=20, sample_rate=22050):
    '''
    INPUT: array (snippets)
    OUTPUT: array (features)
    Extracts features from an array of snippets
    '''
    mfcc = np.mean(librosa.feature.mfcc(y=snippet, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
    return mfcc

def snippet_selector(snippet_features_raw):
    '''
    INPUT:
    OUTPUT:

    '''
    return snippet_features_raw


#########################################
############ Analysis tools #############
#########################################

def snippet_cv(path_full, path_single, splits=1, song_limit=50, artist_limit=2, model=RandomForestClassifier()):
    '''
    Takes in The
    '''
    result = []
    X_full, y_full, song_ids_full = main_engine(path_full, splits=splits, song_limit=song_limit, artist_limit=artist_limit)
    X_single, y_single, song_ids_single = csv_batch_extractor(path_single, song_limit=song_limit, artist_limit=artist_limit)
    sf = StratifiedKFold(n_splits=5, shuffle=True)
    for train, test in sf.split(X_single, y_single):
        X_train = reduce(lambda x, y: np.concatenate((x,y)), X_full[train])
        y_train = reduce(lambda x, y: np.concatenate((x,y)), y_full[train])
        model.fit(X_train, y_train)
        result.append(model.score(X_single[test], y_single[test]))
    return result


if __name__ == '__main__':
    result = snippet_cv('../data/pickles/5s_wo/', '../data/pickles/5s_wo/', artist_limit=None, splits=6, song_limit=100)
    print np.mean(result)