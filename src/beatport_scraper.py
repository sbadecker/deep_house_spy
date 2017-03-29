from bs4 import BeautifulSoup
import requests
import urllib2
import os
from time import time

def track_id_scraper(artist_id):
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
        # import pdb; pdb.set_trace()
        tracks_on_page = len(soup.find_all(class_="buk-track-title")[1:])
        n_tracks += tracks_on_page
        n += 1
    return n_tracks

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
    while len(classes) > 1 and len(inputlist[1]) < max_artists:
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
        inputlist[0] = n
        if n % 10 == 0:
            print n
            print 'Time elapsed', time()-start_time
        n += 1

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

def beatport_downloader(song_ids, directory='./'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for song in song_ids:
        song_url = 'http://geo-samples.beatport.com/lofi/{}.LOFI.mp3'.format(song)
        song_file = urllib2.urlopen(song_url)
        with open(directory+song+'.mp3', 'wb') as f:
            f.write(song_file.read())

def batch_downloader(artist_list):
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

if __name__ == '__main__':
    artists = ['rodriguez-jr/14633', 'sascha-funke/6262', 'and-me/61960',
               'mandingo/21472', 'aero-manyelo/127207', 'eagles-and-butterflies/290967']
    for artist in artists:
        tracks = track_id_scraper(artist)
        beatport_downloader(tracks, './'+artist.split('/')[0]+'/')
