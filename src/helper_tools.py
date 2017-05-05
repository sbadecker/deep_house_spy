import numpy as np
import glob
import os
import shutil
import librosa
# import matplotlib.pyplot as plt
from tempfile import TemporaryFile


#########################################
############  Librosa scripts ###########
#########################################

def plotter(X):
    '''Creates a matplotlib plot.'''
    fig = plt.figure()
    n_plots = len(X)
    for i in range(n_plots):
        plotcode = n_plots*100+10+i+1
        ax = fig.add_subplot(plotcode)
        ax.plot(X[i])
    plt.show()

def heatmap(X, y_labels=[], x_labels=[]):
    '''Creates a heatmap using matplotlib.'''
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
    '''Creates a heatmap of mfccs (or similar features).'''
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(X, x_axis='artist')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


#########################################
#############  File handlers ############
#########################################

def move_done(path_input, path_output, file_extension='*'):
    '''
    Checks which of the files in the input directory are also in the output
    directory. Files from the input directory that also exist in the output
    directory are boing moved to a new directory path_input/done/.
    '''
    input_dirs = glob.glob(path_input+'*.'+file_extension)
    output_dirs = glob.glob(path_output+'*.'+file_extension)

    input_files = [i.split('/')[-1] for i in input_dirs]
    output_files = [i.split('/')[-1] for i in output_dirs]

    done_files = [i for i in input_files if i in output_files]
    done_dirs = [path_input+'done/'+i for i in done_files]

    if not os.path.exists(path_input+'done/'):
        os.makedirs(path_input+'done/')
    for filename in done_files:
        shutil.move(path_input+filename, path_input+'done/'+filename)

def copy_songs(path_songs, path_output, max_artist, file_extension='*'):
    '''Moves song for all artists up to max_artist.'''
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    song_dirs = glob.glob(path_songs+'*.'+file_extension)
    for song in song_dirs:
        songname = song.split('/')[-1]
        artist = int(songname.split('_')[0])
        if artist <= max_artist:
             shutil.copy2(song, path_output)

def csv_exporter(raw_audio_data, path, songdirs):
    '''Exports raw audio data and songdirs to a pickle and csv file.'''
    if not os.path.exists('./raw_data/'):
        os.makedirs('./raw_data/')
    if not os.path.exists('./meta_info/'):
        os.makedirs('./meta_info/')
    np.save('./raw_data/'+path.split('/')[-2], np.array(raw_audio_data), allow_pickle=True)
    with open('./meta_info/'+path.split('/')[-2]+'-metainfo'+'.csv','w') as f:
        for songdir in songdirs:
            f.write(songdir+'\n')

def pickle_exporter(raw_audio_data, path, songdirs):
    '''
    Pickles raw audio data of individual songs and creates individual files with
    the names of the songs (specified in songdirs).
    '''
    if not os.path.exists(path+'output/'):
        os.makedirs(path+'output/')
    for i, song in enumerate(raw_audio_data):
        np.save(path+'output/'+songdirs[i].split('/')[-1][:-4], np.array(song), allow_pickle=True)

def shuffler(X, y):
    '''Returns the identically shuffled version of two arrays.'''
    shuffler = np.array(range(len(X)))
    np.random.shuffle(shuffler)
    X_shuffled = X[shuffler]
    y_shuffled = y[shuffler]
    return X_shuffled, y_shuffled
