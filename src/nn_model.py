import numpy as np
# import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from full_model import main_engine_parallel, csv_batch_extractor
from helper_tools import shuffler


#########################################
############# Keras import ##############
#########################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.datasets import mnist

#########################################
############### Model 1 #################
#########################################

def cnn_model(X_train, X_test, y_train, y_test, n_classes):
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', data_format="channels_first", input_shape=(1,20,44)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=32, epochs=10, verbose=1)
    return model, y_train, y_test

#########################################
############## Predictor ################
#########################################




#########################################
############# Helper tools ##############
#########################################

def train_test_snippets(X, y):
    '''
    Takes in buckets of snippets each representing a song, splitting them into
    training and test sets and then flattening the arrays.
    '''
    X_train_songs, X_test_songs, y_train_songs, y_test_songs = train_test_split(X, y)
    X_train = reduce(lambda x, y: np.concatenate((x,y)), X_train_songs)
    X_test = reduce(lambda x, y: np.concatenate((x,y)), X_test_songs)
    y_train = reduce(lambda x, y: np.concatenate((x,y)), y_train_songs)
    y_test = reduce(lambda x, y: np.concatenate((x,y)), y_test_songs)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    #########################################
    ############# Loading data ##############
    #########################################
    # X = np.load('../data/pickles/incl_features/X_2a_100s_20mfccs.npy')
    # y = np.load('../data/pickles/incl_features/y_2a_100s_20mfccs.npy')

    X, y = main_engine_parallel('../data/pickles/full_songs/', second_snippets=3, song_limit=100, artist_limit=2, n_mfcc=20, full_mfccs=True)

    # X = np.load('../data/pickles/incl_features/X_10a_alls_20mfccs.npy')
    # y = np.load('../data/pickles/incl_features/y_10a_alls_20mfccs.npy')


    X_train, X_test, y_train, y_test = train_test_snippets(X, y)

    X_train = X_train.reshape(X_train.shape[0], 1, 20, 44)
    X_test = X_test.reshape(X_test.shape[0], 1, 20, 44)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train, y_train = shuffler(X_train, y_train)


    #########################################
    ############# Building CNN ##############
    #########################################

    model, y_train_n, y_test_n = cnn_model(X_train, X_test, y_train, y_test, 2)
    print model.evaluate(X_test, y_test_n)
