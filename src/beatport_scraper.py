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

def artist_scraper(inputlist, startpage=1):
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
            inputlist[1].add((artist_link[2], artist_link[3]))
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

if __name__ == '__main__':
    tracks = track_id_scraper('sandrino/100608')
    beatport_downloader(tracks, './sandrino/')
