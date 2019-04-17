import numpy as np
from pathlib import Path
import os 
import shutil
import wave
import contextlib
from pydub import AudioSegment
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import sys



GOOD_FOLDER = "good"
BAD_FOLDER = "bad"
TEMP_GOOD = "temp_good"
TEMP_BAD = "temp_bad"
OUTPUT_FOLDER = "dataset"
TIMING_SUBDIVISION = 1000 # in miliseconds
DATA_AUGMENTATION = True
DATA_AUGMENTED_SHIFT = 500

class ImageGenerator(object):
    def __init__ (self, audio_name,image_output):
        self.audio_name = audio_name
        self.image_output = image_output

    def generate_image(self):
        self.plotstft(self.audio_name, plotpath=self.image_output)

    def plotstft(self,audiopath, binsize=2**10, plotpath=None, colormap="jet"):
        samplerate, samples = wav.read(audiopath)

        s = self.stft(samples, binsize)
        sshow, freq = self.logscale_spec(s, factor=20.0, sr=samplerate)
        ims = 40.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

        timebins, freqbins = np.shape(ims)

        #plt.figure(figsize=(15, 7.5))
        #plt.tick_params(axis='both', which='both',length=0,labelbottom=False, labelleft=False)
        #plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        data = np.transpose(ims)
        sizes = np.shape(data)
        height = float(sizes[0])
        width = float(sizes[1])
         
        fig = plt.figure(figsize=(10, 10))
        #fig.set_size_inches(width/height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(data, origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        

        if plotpath:
            #plt.savefig(plotpath, bbox_inches="tight",transparent=True)
            plt.savefig(plotpath, dpi = height) 
        else:
            plt.show()

        #plt.clf()
        plt.close()


        return ims
    
    @staticmethod
    def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
        win = window(frameSize)
        hopSize = int(frameSize - np.floor(overlapFac * frameSize))

        # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
        samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
        # cols for windowing
        cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
        # zeros at end (thus samples can be fully covered by frames)
        samples = np.append(samples, np.zeros(frameSize))

        frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
        frames *= win

        return np.fft.rfft(frames)    

    @staticmethod   
    def logscale_spec(spec, sr=44100, factor=20.):
        timebins, freqbins = np.shape(spec)

        scale = np.linspace(0, 1, freqbins) ** factor
        scale *= (freqbins-1)/max(scale)
        scale = np.unique(np.round(scale))

        # create spectrogram with new freq bins
        newspec = np.complex128(np.zeros([timebins, len(scale)]))
        for i in range(0, len(scale)):        
            if i == len(scale)-1:
                newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
            else:        
                newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

        # list center freq of bins
        allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
        freqs = []
        for i in range(0, len(scale)):
            if i == len(scale)-1:
                freqs += [np.mean(allfreqs[int(scale[i]):])]
            else:
                freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

        return newspec, freqs


def get_time_audio(file_name):
    length = 0
    with contextlib.closing(wave.open(file_name,'r')) as f: 
        frames = f.getnframes()
        rate = f.getframerate()
        length = frames / float(rate)    
        return length *1000

def subdive_audio(file_name,partitions, output_folder):
    for values in partitions:
        t1 = values[0] #Works in milliseconds
        t2 = values[1]
        newAudio = AudioSegment.from_wav(file_name)
        newAudio = newAudio[t1:t2]
        filename = file_name.split("/")[1]
        values_name = filename.split('.')
        newAudio.export( output_folder+ "/"+ values_name[0] + "_%s_%s.wav" % (t1,t2), format="wav") #Exports to a wav file in the current path.

def process_file(file_name, output_folder):
    # get time
    time = get_time_audio(file_name)
    parts = int(time)/TIMING_SUBDIVISION
    number = int(round(parts,0))
    partitions = []
    for i in range(0, number):
            partitions.append([TIMING_SUBDIVISION*(i),TIMING_SUBDIVISION*(i+1)])
    
    if DATA_AUGMENTATION:
        number_times = TIMING_SUBDIVISION / DATA_AUGMENTED_SHIFT
        for i in range(0, number):
            for j in range(1, int(round(number_times))):
                base_a = (TIMING_SUBDIVISION*(i))+DATA_AUGMENTED_SHIFT*j
                base_b = (TIMING_SUBDIVISION*(i+1))+DATA_AUGMENTED_SHIFT*j
                if base_a <= int(time) and base_b <= int(time):
                    partitions.append([base_a,base_b])
    
    subdive_audio(file_name, partitions, output_folder)
    


def main():
    # create folders
    os.mkdir(TEMP_GOOD)
    os.mkdir(TEMP_BAD)
    os.mkdir(OUTPUT_FOLDER)
    os.mkdir(OUTPUT_FOLDER+"/"+GOOD_FOLDER)
    os.mkdir(OUTPUT_FOLDER+"/"+BAD_FOLDER)
    # loop over folders and create subdivision time
    for subdir, dirs, files in os.walk(GOOD_FOLDER):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            process_file(filepath, TEMP_GOOD)

    for subdir, dirs, files in os.walk(BAD_FOLDER):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            process_file(filepath, TEMP_BAD)


    # convert sounds into images
    for subdir, dirs, files in os.walk(TEMP_GOOD):
        file_number = 0
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            a = ImageGenerator(filepath, OUTPUT_FOLDER+"/"+GOOD_FOLDER+"/"+str(file_number))
            a.generate_image()
            file_number += 1

    for subdir, dirs, files in os.walk(TEMP_BAD):
        file_number = 0
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            a = ImageGenerator(filepath,  OUTPUT_FOLDER+"/"+BAD_FOLDER+"/"+str(file_number))
            a.generate_image()
            file_number += 1


    #a = ImageGenerator("Normal_Breath_Sound.wav", "Normal_Breath_Sound.jpg")
    #a.generate_image()

    shutil.rmtree(TEMP_GOOD)
    shutil.rmtree(TEMP_BAD)











if __name__ == "__main__":
    if "clean" in str(sys.argv):
        try:
            shutil.rmtree(OUTPUT_FOLDER)
        except Exception as e:
            raise
        
    else:
        main()