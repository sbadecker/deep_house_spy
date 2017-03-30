import numpy as np
import glob
import os
import librosa
import multiprocessing
from functools import partial
from helper_tools import pickle_exporter, move_done
from full_model import snippet_feature_extractor

#########################################
########### Audio extraction ############
#########################################

def parallel_audio_extractor(path, format='mp3', duration=None, offset=0.0, song_limit=None, sample_rate=22050, pool_size=7, batch_size=24):
    '''
    Takes in the path to audio files (mp3) and loads them as floating time series.
    It calls the pickle_exporter which stores the audio and meta data as individual
    picklefiles.

    Can run use multiple cores (specified by pool_size).
    '''
    songdirs = glob.glob(path+'*.'+format)
    n_chunks = len(songdirs)/batch_size
    splitter = [batch_size*i for i in range(1,n_chunks)]
    songdir_chunks = np.split(songdirs, splitter)
    for chunk in songdir_chunks:
        raw_audio_data = []
        pool = multiprocessing.Pool(processes=pool_size)
        X = pool.map(partial(librosa.load, duration=duration, offset=offset, sr=sample_rate), chunk)
        pool.close()
        pool.join()
        sr = X[0][1]
        for song in X:
            raw_audio_data.append(song[0])
        pickle_exporter(raw_audio_data, path, chunk)


#########################################
########### Feature extraction ##########
#########################################

def parallel_feature_extractor(path, n_mfcc=20, second_snippets=1, pool_size=7, sample_rate=22050, i=0):
    '''
    Takes in a path and loads in all .npy files included in this directory. It
    the splits the files in snippets and extracts the MFCCs from saves them in
    a new directory as npys.
    '''
    raw_songdirs = glob.glob(path+'*.npy')[i:]
    pool = multiprocessing.Pool(processes=pool_size)
    X = pool.map(partial(parallel_feature_child, path=path, frames=sample_rate*second_snippets, n_mfcc=n_mfcc), raw_songdirs)
    pool.close()
    pool.join()

def parallel_feature_child(song, path, frames, n_mfcc):
    if not os.path.exists(path+'features_extracted/'):
        os.makedirs(path+'features_extracted/')
    X_song = []
    print song
    raw_audio_song = np.load(song)
    total_splits = raw_audio_song.shape[0]/frames
    splits = [n*frames for n in range(1, 120)]
    snippets = np.split(raw_audio_song, splits)
    snippets = snippets[:-1]
    for snippet in snippets:
        snippet_features_raw = snippet_feature_extractor(snippet, n_mfcc=n_mfcc, full_mfccs=True)
        X_song.append(snippet_features_raw)
    np.save(path+'features_extracted/'+song.split('/')[-1][:-4], X_song, allow_pickle=True)


#########################################
########### Feature extraction ##########
#########################################

def song_combiner(path):
    '''
    Takes in a path with npy files that from the parallel_feature_extractor and
    combines those into X  (n_songs x n_snippets x n_mfccs x time_frames) and
    y (n_songs x n_snippets).
    '''
    X = []
    y = []
    song_dirs = glob.glob(path+'*.npy')
    for i, song in enumerate(song_dirs):
        X_song = np.load(song)
        artist = [int(song.split('/')[-1].split('_')[0])]
        y_song = artist * X_song.shape[0]
        X.append(X_song)
        y.append(y_song)
        print i
    return np.array(X), np.array(y)


if __name__ == '__main__':
    # X, y = song_combiner('../data/100_artists/features_extracted/')
    pass
