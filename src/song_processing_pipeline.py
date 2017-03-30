import numpy as np
import glob
import os
import librosa
import multiprocessing
from functools import partial
from helper_tools import pickle_exporter

def parallel_audio_extractor(path, format='mp3', duration=None, offset=0.0, song_limit=None, sample_rate=22050, pool_size=7, batch_size=20):
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
        import pdb; pdb.set_trace()
        sr = X[0][1]
        for song in X:
            raw_audio_data.append(song[0])
        # import pdb; pdb.set_trace()
        pickle_exporter(raw_audio_data, path, chunk)


if __name__ == '__main__':
    parallel_audio_extractor('../data/100_artists/mp3s/')
