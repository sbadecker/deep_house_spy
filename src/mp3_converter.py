# from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from scipy import fft, arange, ifft
from numpy import sin, linspace, pi
# from scipy.io.wavfile import read,write
from pydub import AudioSegment
# import pydub
import glob

# Getting beatport previews: http://geo-samples.beatport.com/lofi/9040275.LOFI.mp3



def mp3_to_wav(path):
    songdirs = glob.glob(path+'*.mp3')
    for song in songdirs:
        sound = AudioSegment.from_mp3(song)
        sound.export(path+'wav/'+song.split('/')[-1][:-4]+'.wav', format="wav")


if __name__ == '__main__':
    # songdirs = glob.glob('../data/songs/aero-manyelo/'+'*.mp3')
    # for song in songdirs:
    #     sound = AudioSegment.from_mp3(song)
    #     sound.export('../data/songs/aero-manyelo/wav/'+song.split('/')[-1][:-4]+'.wav', format="wav")

    mp3_to_wav('../data/songs/aero-manyelo/')
