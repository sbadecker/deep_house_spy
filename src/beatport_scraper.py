from bs4 import BeautifulSoup
import glob
import requests
import urllib2
import os
import numpy as np
from time import time

#########################################
############# Track scraper #############
#########################################

def tracklist_creator(artist_data):
    '''
    Takes in a list of lists with artist_name and artist_ids. Loops through
    all artists, scrapes their songs with the track_data_scraper and removes
    duplicate songs. It also assigns a class to each artist and returns an
    artist_index with the artist_class, artist_id and number of songs.
    '''
    artist_index = []
    song_list = []
    artist_class = 0
    for artist_name, artist_id in artist_data:
        number_of_songs = 0
        song_data = track_data_scraper(artist_class, artist_name, artist_id)
        for song in song_data:
            if song[-1] not in [row[-1] for row in song_list]:
                song_list.append(song)
                number_of_songs += 1
        artist_index.append([str(artist_class), artist_name, str(number_of_songs)])
        artist_class += 1
    return song_list, artist_index

def track_data_scraper(artist_class, artist_name, artist_id):
    '''
    Takes in an artist name and artist id and scrapes the ids and names from all
    songs associated with that artist and returns those as a list.
    '''
    n = 1
    song_data = []
    classes = [0,0]
    while len(classes) > 1:
        url = 'https://www.beatport.com/artist/{}/{}/tracks?page={}'.format(artist_name, artist_id, n)
        result = requests.get(url)
        content = result.content
        soup = BeautifulSoup(content, 'html.parser')
        classes = soup.find_all(class_="buk-track-title")[1:]
        for song in classes:
            song_name = song.find('a').attrs['href'].split('/')[-2]
            song_id = song.find('a').attrs['href'].split('/')[-1]
            song_data.append([str(artist_class), artist_name, artist_id, song_name, song_id])
        n += 1
    return song_data

def track_id_scraper_old(artist_id):
    n = 1
    song_ids = []
    classes = [0,0]
    while len(classes) > 1:
        url = 'https://www.beatport.com/artist/{}/tracks?page={}'.format(artist_id, n)
        result = requests.get(url)
        content = result.content
        soup = BeautifulSoup(content, 'html.parser')
        classes = soup.find_all(class_="buk-track-title")[1:]
        for title in classes:
            song_ids.append(title.find('a').attrs['href'].split('/')[-1])
        n += 1
    return song_ids

def track_checker(artist_name, artist_id):
    '''
    Takes in the artist id and returns the number of songs that this artist has on beatport.
    '''
    n_tracks = 0
    tracks_on_page = 1
    n = 1
    while tracks_on_page > 0:
        url = 'https://www.beatport.com/artist/{}/{}/tracks?page={}'.format(artist_name, artist_id, n)
        result = requests.get(url)
        content = result.content
        soup = BeautifulSoup(content, 'html.parser')
        tracks_on_page = len(soup.find_all(class_="buk-track-title")[1:])
        n_tracks += tracks_on_page
        n += 1
    return n_tracks

def list_cleaner(song_list):
    song_list_cleaned = []
    for song in song_list:
        if song[-2] not in [row[-2] for row in song_list_cleaned]:
            song_list_cleaned.append(song)
    return song_list_cleaned

def list_reducer(song_list, max_per_class, max_class):
    song_list = np.array(song_list)
    np.random.shuffle(song_list)
    song_list_reduced = song_list[1:2]
    for i in range(max_class):
        reduced_chunk = song_list[song_list[:,0]==str(i)][:max_per_class]
        # import pdb; pdb.set_trace()
        song_list_reduced = np.append(song_list_reduced, reduced_chunk, axis=0)
    return song_list_reduced[1:]


#########################################
############ Artist scraper #############
#########################################

def artist_scraper(inputlist, startpage=1, min_songs=None, max_artists=10000):
    '''
    INPUT: list with the following form: [int, set()]
    OUTPUT: none
    The int on index 0 of the list will be filled with the number of the latest
    page that has been scraped. The set will be updated with the artists. If the
    process is stopped, it can be resumed at a later point in time.
    '''
    n = startpage
    classes = [0,0]
    start_time = time()
    while len(classes) > 1:
        url = 'https://www.beatport.com/genre/deep-house/12/tracks?per-page=150&page={}'.format(n)
        result = requests.get(url)
        content = result.content
        soup = BeautifulSoup(content, 'html.parser')
        classes = soup.find_all(class_='buk-track-artists')
        for artist in classes[1:]:
            artist_link = artist.find('a').attrs['href'].split('/')
            artist_name = artist_link[2]
            artist_id = artist_link[3]
            if min_songs > 0:
                if track_checker(artist_name, artist_id) >= min_songs:
                    inputlist[1].add((artist_name, artist_id))
            else:
                inputlist[1].add((artist_name, artist_id))
            if len(inputlist[1]) == max_artists:
                return None
        inputlist[0] = n
        if n % 2 == 0:
            print n
            print 'Time elapsed', time()-start_time
        n += 1

def beatport_url_artist_scraper(beatport_url, inputlist, min_songs=None):
    '''
    INPUT: list with the following form: [int, set()]
    OUTPUT: none
    The int on index 0 of the list will be filled with the number of the latest
    page that has been scraped. The set will be updated with the artists. If the
    process is stopped, it can be resumed at a later point in time.
    '''
    classes = [0,0]
    start_time = time()
    url = beatport_url
    result = requests.get(url)
    content = result.content
    soup = BeautifulSoup(content, 'html.parser')
    classes = soup.find_all(class_='buk-track-artists')
    for artist in classes[1:]:
        artist_link = artist.find('a').attrs['href'].split('/')
        artist_name = artist_link[2]
        artist_id = artist_link[3]
        if min_songs > 0:
            if track_checker(artist_name, artist_id) >= min_songs:
                inputlist[1].add((artist_name, artist_id))
        else:
            inputlist[1].add((artist_name, artist_id))

#########################################
############## Downloader ###############
#########################################

def beatport_downloader(song_list, directory='./'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for artist_class, artist_name, artist_id, song_name, song_id in song_list:
        song_url = 'http://geo-samples.beatport.com/lofi/{}.LOFI.mp3'.format(song_id)
        song_file = urllib2.urlopen(song_url)
        song_file_name = artist_class+'_'+artist_name+'_'+artist_id+'_'+song_name+'_'+song_id+'.mp3'
        with open(directory+song_file_name, 'wb') as f:
            f.write(song_file.read())

def batch_downloader_old(artist_list):
    '''
    INPUT: list
    OUT: None
    Takes in a list of artiest names and ids, and
    1. Scrapes all of their songs with the track_id_scraper
    2. Downloads all their songs with the beatport_downloader
    '''
    for artist in artist_list:
        tracks = track_id_scraper(artist[0]+'/'+artist[1])
        beatport_downloader(tracks, './'+artist[0]+'/')

def download_checker(path, song_list):
    downloaded = glob.glob(path+'*.mp3')
    downloaded = [i.split('/')[-1] for i in downloaded]
    downloaded = [i[:-4].split('_') for i in downloaded]
    download_list = [list(line) for line in song_list if list(line) not in downloaded]
    return download_list

#########################################
############### CSV saver ###############
#########################################

def artist_saver(inputlist, outputfile):
    '''
    INPUT: list, outputfile
    OUTPUT: none
    Takes the list of the artist_scraper and saves it as a csv file
    '''
    with open(outputfile, 'wb') as f:
        f.write(str(inputlist[0])+'\n')
        for line in inputlist[1]:
            f.write(','.join(line)+'\n')


if __name__ == '__main__':
    song_list = np.loadtxt('../data/100_artists/_old/song_list_cleaned.csv', dtype=str, delimiter=',')
    download_list = download_checker('../data/100_artists/mp3s/', song_list)

    beatport_downloader(download_list, '../data/100_artists/mp3s2/')
