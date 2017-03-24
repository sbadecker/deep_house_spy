from scipy import fft, arange, ifft
from numpy import sin, linspace, pi
from pydub import AudioSegment
import glob

def mp3_to_wav(path):
    songdirs = glob.glob(path+'*.mp3')
    for song in songdirs:
        sound = AudioSegment.from_mp3(song)
        sound.export(path+'wav/'+song.split('/')[-1][:-4]+'.wav', format="wav")


if __name__ == '__main__':
    # mp3_to_wav('../data/songs/aero-manyelo/')
