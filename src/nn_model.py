import numpy as np
# import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from full_model import main_engine_parallel, csv_batch_extractor
from helper_tools import shuffler
from scipy.stats import mode
from song_processing_pipeline import song_combiner

#########################################
############# Keras import ##############
#########################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.utils import np_utils

#########################################
############### Model 1 #################
#########################################

def cnn_model(X_train, X_test, y_train, y_test, n_classes):
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_first", input_shape=(1,20,44)))
    model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_first"))
    model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_first"))

    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_first"))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=32, epochs=10, verbose=1)
    return model, y_train, y_test

#########################################
############### Model 2 #################
#########################################

def cnn_model_2(X_train, X_test, y_train, y_test, n_classes):
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    model = Sequential()

    model.add(ZeroPadding2D(padding=2, data_format="channels_first", input_shape=(1, 20, 44)))
    model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_first"))
    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(32, (3, 3), data_format="channels_first", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_first"))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(64, (3, 3), data_format="channels_first", activation='relu'))
    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(64, (3, 3), data_format="channels_first", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), data_format="channels_first"))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(128, (3, 3), data_format="channels_first", activation='relu'))
    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(128, (3, 3), data_format="channels_first", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), data_format="channels_first"))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(256, (3, 3), activation='relu', data_format="channels_first"))
    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(256, (3, 3), activation='relu', data_format="channels_first"))
    model.add(GlobalAveragePooling2D(data_format="channels_first"))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=32, epochs=20, verbose=1)

    return model, y_train, y_test

def cnn_model_2_full(X, y, n_classes):
    y = np_utils.to_categorical(y, n_classes)

    model = Sequential()

    model.add(ZeroPadding2D(padding=2, data_format="channels_first", input_shape=(1, 20, 44)))
    model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_first"))
    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(32, (3, 3), data_format="channels_first", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_first"))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(64, (3, 3), data_format="channels_first", activation='relu'))
    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(64, (3, 3), data_format="channels_first", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), data_format="channels_first"))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(128, (3, 3), data_format="channels_first", activation='relu'))
    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(128, (3, 3), data_format="channels_first", activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), data_format="channels_first"))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(256, (3, 3), activation='relu', data_format="channels_first"))
    model.add(ZeroPadding2D(padding=2, data_format="channels_first"))
    model.add(Conv2D(256, (3, 3), activation='relu', data_format="channels_first"))
    model.add(GlobalAveragePooling2D(data_format="channels_first"))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(X, y,
              batch_size=32, epochs=20, verbose=1)

    return model, y


#########################################
############## Predictors ###############
#########################################

def cnn_predict(model, song, reshape=True):
    '''
    Takes in a CCN model and an array with snippets of a song (shape: n_snippets,
    buckets, frames) and runs a prediction on them. The snippets of each song
    are predicted individually. The prediction is then averaged.
    '''
    if reshape:
        song = song.reshape(song.shape[0],1,song.shape[1],song.shape[2])
    proba_snippets = model.predict_proba(song, verbose=0)
    proba_classes = np.mean(proba_snippets, axis=0)
    top_prediction = proba_classes.argmax()
    return top_prediction

def top_n_predict(model, song, n_artists, reshape=True):
    '''
    Takes in a CCN model and an array with snippets of a song (shape: n_snippets,
    buckets, frames) and runs a prediction on them. The snippets of each song
    are predicted individually. The softmax arrays are then summed averaged for
    along the classes and the top n_artists classes are returned including the probabilities.
    '''
    if reshape:
        song = song.reshape(song.shape[0],1,song.shape[1],song.shape[2])
    prediction = model.predict_proba(song, verbose=0)
    proba_classes = np.mean(prediction, axis=0)
    top_n_artists = (proba_classes*-1).argsort()[:n_artists]
    return top_n_artists


#########################################
############ Analysis tools #############
#########################################

def ensemble_accuracy(model, X_test, y_test, start=None, end=None):
    result = []
    for i, song in enumerate(X_test):
        prediction = cnn_predict(model, song[start:end], reshape=True)
        correct = (prediction == y_test[:,1][i])*1.
        result.append(correct)
    return result

def top_n_accuracy(model, X_test, y_test, n_artists, start=None, end=None):
    result = []
    for i, song in enumerate(X_test):
        prediction = top_n_predict(model, song[start:end], n_artists, reshape=True)
        correct = (y_test[:,1][i] in prediction)*1.
        result.append(correct)
    return result

#########################################
######### Train test splitter ###########
#########################################

def train_test_snippets(X, y, untouched=True):
    '''
    Takes in buckets of snippets each representing a song, splitting them into
    training and test sets and then flattening the arrays.
    '''
    X_train_songs, X_test_songs, y_train_songs, y_test_songs = train_test_split(X, y)
    X_test_untouched = X_test_songs
    y_test_untouched = y_test_songs
    X_train = np.concatenate(X_train_songs)
    X_test = np.concatenate(X_test_songs)
    y_train = np.concatenate(y_train_songs)
    y_test = np.concatenate(y_test_songs)
    if untouched:
        return X_train, X_test, y_train, y_test, X_test_untouched, y_test_untouched
    else:
        return X_train, X_test, y_train, y_test

def stratified_split(X, y, untouched=True):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.25)
    for train_index, test_index in sss.split(X, y[:,1]):
        X_train_songs = X[train_index]
        X_test_songs = X[test_index]
        y_train_songs = y[train_index]
        y_test_songs = y[test_index]
    X_test_untouched = X_test_songs
    y_test_untouched = y_test_songs
    X_train = np.concatenate(X_train_songs)
    X_test = np.concatenate(X_test_songs)
    y_train = np.concatenate(y_train_songs)
    y_test = np.concatenate(y_test_songs)
    if untouched:
        return X_train, X_test, y_train, y_test, X_test_untouched, y_test_untouched
    else:
        return X_train, X_test, y_train, y_test

#########################################
############## Data loader ##############
#########################################

def full_data_loader(xy_directory):
    X, y = song_combiner(xy_directory)

    X = np.concatenate(X)
    y = np.concatenate(y)

    X = X.reshape(X.shape[0], 1, 20, 44)
    X = X.astype('float32')
    return X, y

def train_test_data_loader(X_filename, y_filename):
    X = np.load('../data/pickles/incl_features/X_10a_alls_20mfccs.npy')
    y = np.load('../data/pickles/incl_features/y_10a_alls_20mfccs.npy')

    X_train, X_test, y_train, y_test, X_test_untouched, y_test_untouched = stratified_split(X, y)

    X_train = X_train.reshape(X_train.shape[0], 1, 20, 44)
    X_test = X_test.reshape(X_test.shape[0], 1, 20, 44)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    return X_train, X_test, y_train, y_test, X_test_untouched, y_test_untouched
