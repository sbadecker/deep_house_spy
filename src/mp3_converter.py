from pylab import plot, show, title, xlabel, ylabel, subplot, savefig
from scipy import fft, arange, ifft
from numpy import sin, linspace, pi
from scipy.io.wavfile import read,write
from pydub import AudioSegment
import pydub

# Getting beatport previews: http://geo-samples.beatport.com/lofi/9040275.LOFI.mp3

sound = AudioSegment.from_mp3("/path/to/file.mp3")
sound.export("/output/path", format="wav")

def mp3_to_wav(file_in):
    sound = pydub.AudioSegment.from_mp3(filename)
    file_out = file_in[:-4]+'.wav'
    sound.export(file_out)
