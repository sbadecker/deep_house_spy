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
from sklearn.cluster import KMeans
import multiprocessing
from functools import partial


#########################################
############# Main engine ###############
#########################################

def main_engine(path, second_snippets=1, song_limit=None, artist_limit=None, n_mfcc=8, force_full_song=False):
    '''
    INPUT: Directory of Artist directories with pickles in them
    OUTPUT: X, y, song_ids

    Loads in all songs from artists in the specified path, and handles the
    splitting, feature extraction and snippet selection process.
    '''
    start = time()
    X = []
    y = []
    song_ids = []
    frames = second_snippets * 22050
    raw_artistfiles = sorted(glob.glob(path+'raw_data/'+'*.npy'))[:artist_limit]
    meta_artistfiles = sorted(glob.glob(path+'meta_info/'+'*.csv'))[:artist_limit]
    for i, raw_artist in enumerate(raw_artistfiles):
        raw_audio_data = np.load(raw_artist)[:song_limit]
        songdirs = np.loadtxt(meta_artistfiles[i], dtype=str)[:song_limit]
        for j, song in enumerate(raw_audio_data):
            X_song = []
            y_song = []
            song_ids_song = []
            total_splits = song.shape[0]/frames
            splits = [n*frames for n in range(1, 120)]
            snippets = np.split(song, splits)
            if not force_full_song:
                snippets = snippets[:-1]
            for snippet in snippets:
                snippet_features_raw = snippet_feature_extractor(snippet, n_mfcc=n_mfcc)
                X_song.append(snippet_features_raw)
                y_song.append(i)
                song_ids_song.append(songdirs[j])
            X_song = snippet_selector(X_song)
            X.append(X_song)
            y.append(y_song[:len(X_song)])
            song_ids.append(song_ids_song[:len(X_song)])
            if j != 0 and (j+1) % 10 == 0:
                print 'Artist %d, %d songs done' % (i, j+1)
    print 'Total runtime: ', time()-start
    return np.array(X), np.array(y), np.array(song_ids)


#########################################
############# Parallelized ##############
#########################################

def main_engine_parallel(path, second_snippets=1, song_limit=None, artist_limit=None, n_mfcc=8, pool_size=8, full_mfccs=False, force_full_song=False):
    '''
    Same functionality as main_engine but can use multiple cores.
    '''
    start = time()
    X = []
    y = []
    song_ids = []
    raw_artistfiles = sorted(glob.glob(path+'raw_data/'+'*.npy'))[:artist_limit]
    meta_artistfiles = sorted(glob.glob(path+'meta_info/'+'*.csv'))[:artist_limit]
    for i, raw_artist in enumerate(raw_artistfiles):
        X_artist = []
        y_artist = []
        raw_audio_data = np.load(raw_artist)[:song_limit]
        songdirs = np.loadtxt(meta_artistfiles[i], dtype=str)[:song_limit]
        pool = multiprocessing.Pool(processes=pool_size)
        data_artist = pool.map(partial(parallel_child, i=i, frames=22050*second_snippets, n_mfcc=n_mfcc, full_mfccs=full_mfccs), raw_audio_data)
        pool.close()
        pool.join()
        for song in data_artist:
            X_artist.append(song[0])
            y_artist.append(song[1])
        X.append(np.array(X_artist))
        y.append(np.array(y_artist))
        print 'Artist %d done' % (i+1)
    X = reduce(lambda x,y: np.append(x,y, axis=0), X)
    y = reduce(lambda x,y: np.append(x,y, axis=0), y)
    print 'Total runtime: ', time()-start
    return X, y

def parallel_child(song, i, n_mfcc, frames=22050, full_mfccs=False, force_full_song=False):
    X_song = []
    y_song = []
    total_splits = song.shape[0]/frames
    splits = [n*frames for n in range(1, 120)]
    snippets = np.split(song, splits)
    if not force_full_song:
        snippets = snippets[:-1]
    for snippet in snippets:
        snippet_features_raw = snippet_feature_extractor(snippet, n_mfcc=n_mfcc, full_mfccs=full_mfccs)
        X_song.append(snippet_features_raw)
    X_song = snippet_selector(X_song)
    y_song = np.ones(len(X_song)) * i
    return np.array(X_song), y_song


#########################################
########### Snippet handlers ############
#########################################

def snippet_feature_extractor(snippet, n_mfcc=20, sample_rate=22050, full_mfccs=False):
    '''
    INPUT: array (snippets)
    OUTPUT: array (features)

    Extracts features from an array of snippets.
    '''
    if full_mfccs:
        mfcc = librosa.feature.mfcc(y=snippet, sr=sample_rate, n_mfcc=n_mfcc)
    else:
        mfcc = np.mean(librosa.feature.mfcc(y=snippet, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
    return mfcc

def snippet_selector(snippet_features_raw):
    '''
    Is meant to intelligently select snippets that are representative for a song.
    For now just returns the input.
    '''
    return snippet_features_raw


#########################################
############ Analysis tools #############
#########################################

def snippet_cv(path_full, path_single, frames=22050, song_limit=50, artist_limit=2, n_mfcc=8, model=RandomForestClassifier()):
    result = []
    X_full, y_full = main_engine_parallel(path_full, frames=frames, song_limit=song_limit, artist_limit=artist_limit, n_mfcc=n_mfcc)
    X_single, y_single, song_ids_single = csv_batch_extractor(path_single, song_limit=song_limit, artist_limit=artist_limit, n_mfcc=n_mfcc)
    sf = StratifiedKFold(n_splits=5, shuffle=True)
    for train, test in sf.split(X_single, y_single):
        X_train = reduce(lambda x, y: np.concatenate((x,y)), X_full[train])
        y_train = reduce(lambda x, y: np.concatenate((x,y)), y_full[train])
        model.fit(X_train, y_train)
        result.append(model.score(X_single[test], y_single[test]))
    return result
    

if __name__ == '__main__':
    pass
