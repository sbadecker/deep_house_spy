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

def main_engine(path, second_snippets=1, song_limit=None, artist_limit=None, n_mfcc=8):
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
            splits = [i*frames for i in range(1, 120)]
            snippets = np.split(song, splits)
            if len(snippets[-1]) != len(snippets[0]):
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

def main_engine_parallel(path, second_snippets=1, song_limit=None, artist_limit=None, n_mfcc=8, pool_size=8, full_mfccs=False):
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


def parallel_child(song, i, n_mfcc, frames=22050, full_mfccs=False):
    X_song = []
    y_song = []
    # song_ids_song = []
    total_splits = song.shape[0]/frames
    splits = [i*frames for i in range(1, 120)]
    snippets = np.split(song, splits)
    if len(snippets[-1]) != len(snippets[0]):
        snippets = snippets[:-1]
    for snippet in snippets:
        snippet_features_raw = snippet_feature_extractor(snippet, n_mfcc=n_mfcc, full_mfccs=full_mfccs)
        X_song.append(snippet_features_raw)
        # y_song.append(i)
        # song_ids_song.append(songdirs[j])
    X_song = snippet_selector(X_song)
    y_song = np.ones(len(X_song)) * i
    # song_ids.append(song_ids_song[:len(X_song)])
    return np.array(X_song), y_song


#########################################
########### Snippet handlers ############
#########################################

def snippet_feature_extractor(snippet, n_mfcc=20, sample_rate=22050, full_mfccs=False):
    '''
    INPUT: array (snippets)
    OUTPUT: array (features)
    Extracts features from an array of snippets
    '''
    if full_mfccs:
        mfcc = librosa.feature.mfcc(y=snippet, sr=sample_rate, n_mfcc=n_mfcc)
    else:
        mfcc = np.mean(librosa.feature.mfcc(y=snippet, sr=sample_rate, n_mfcc=n_mfcc).T,axis=0)
    return mfcc

def snippet_selector(snippet_features_raw):
    '''
    INPUT:
    OUTPUT:
    '''
    # model = KMeans(n_clusters=50)
    # model.fit(snippet_features_raw)
    # return model.cluster_centers_
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
    # X,  y, songs = main_engine('../data/pickles/5s_wo/', splits=20, song_limit=1, artist_limit=1)

    X, y = main_engine_parallel('../data/pickles/full_songs/', second_snippets=1, song_limit=None, artist_limit=10, n_mfcc=20, full_mfccs=True)

    ### pickling
    np.save('../data/pickles/10a_alls_20mfccs', [X, y], allow_pickle=True)

    # X, y, z = main_engine('../data/pickles/full_songs/', splits=120, song_limit=20, artist_limit=2, n_mfcc=8)

    # result = snippet_cv('../data/pickles/full_songs/', '../data/pickles/5s_wo/', splits=120, song_limit=100, artist_limit=2, n_mfcc=20)
    # print 'Middle 5s on 120 snippets:',np.mean(result)

    # result = snippet_cv('../data/pickles/full_songs/', '../data/pickles/5s_wo/', splits=120, song_limit=None, artist_limit=None, n_mfcc=8)
    # print 'Middle 5s on 120 snippets, all artist:',np.mean(result)
