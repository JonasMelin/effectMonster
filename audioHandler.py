from definitions import Definitions as defs
import random
from scipy.io import wavfile
import os
import numpy as np
import copy
import math
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# ############################################################################
# CLASS to handle reading of wav files from disk
# ############################################################################
class AudioHandler:

    SIXTEENBITMAX = 32768
    UINTMAX=255

    # The maximum value and center weight of downscaled data. Note that
    # the value 1.0 is theoretical max of sigmoid!
    # Example: K = 0.95, center = 0.5 --> amplitude from 0.05 -> 0.95 centered around 0.5 (typicall sigmoid activation)
    # Example: K = 1000, center = 600 --> amplitude from 100 -> 1100 centered around 600 (typical RELU activation)
    K = 2.0
    center = 0


    # ############################################################################
    # construct
    # ############################################################################
    def __init__(self):
        pass

    # ############################################################################
    # Reads all sound files in dir and returns the data in a dictionary
    # ############################################################################
    def readAllFilesInDir(self, dir):

        try:
            allData = []
            files = os.listdir(dir)
            for nextFile in files:
                completePath = os.path.join(dir, nextFile)
                if os.path.isfile(completePath) and ".wav" in nextFile.lower():

                    sampleRate, data = wavfile.read(completePath)
                    numberOfChannels = len(data.shape)
                    sampleCount = data.shape[0]
                    trackLengthSec = sampleCount/sampleRate
                    print(f"  - Reading \"{completePath}\" ({sampleRate/1000}kHz, {numberOfChannels} channel(s), length: {trackLengthSec:.1f}s)...")

                    for nextChannel in range(numberOfChannels):

                        if numberOfChannels is 1:
                            audio = data
                        elif numberOfChannels is 2:
                            audio = data[:,nextChannel]
                        else:
                            print(f"Unsupported format! {nextFile} - numberOfChannels {numberOfChannels}")
                            continue

                        if data.dtype is np.dtype('int16'):
                            #Good!
                            pass
                        elif data.dtype is np.dtype('uint8'):
                            audio = self.convertUINT8audioToINT16(audio)
                        else:
                            print(f"UNSUPPORTED data bit depth! {nextFile} - datatype {data.dtype}")
                            continue

                        allData.append({
                            "sampleRate": sampleRate,
                            "data": audio,
                            "scaledData" : self.downScaleAudio(audio),
                            "frequencySpectrum": None,
                            "spectrumLen": 0,
                            "fileName": nextFile,
                            "userDefinedName": "",
                            "sampleCount": sampleCount,
                            "trackLengthSec": trackLengthSec,
                            "scaling": 1.0
                            })

            print(f"Done reading sound files from disk. Read {len(allData)} channels")

            return allData

        except Exception as ex:
            print(f"Failed to read some wav files... {ex}")
            return allData

    # ############################################################################
    # ...
    # ############################################################################
    def addFFTToSound(self, sound):

        if(sound["frequencySpectrum"] is not None):
            # Already a freq spectrum in this sound...
            return

        fftArray = fft(sound["data"])

        numUniquePoints = math.ceil((sound["sampleCount"] + 1) / 2.0)
        fftArray = fftArray[0:numUniquePoints]

        # FFT contains both magnitude and phase and given in complex numbers in real + imaginary parts (a + ib) format.
        # By taking absolute value , we get only real part

        fftArray = abs(fftArray)

        # Scale the fft array by length of sample points so that magnitude does not depend on
        # the length of the signal or on its sampling frequency

        fftArray = fftArray / float(sound["sampleCount"])

        # FFT has both positive and negative information. Square to get positive only
        fftArray = fftArray ** 2

        # Multiply by two (research why?)
        # Odd NFFT excludes Nyquist point
        if sound["sampleCount"] % 2 > 0:  # we've got odd number of points in fft
            fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2

        else:  # We've got even number of points in fft
            fftArray[1:len(fftArray) - 1] = fftArray[1:len(fftArray) - 1] * 2


        sound["freqArray"] = np.arange(0, numUniquePoints, 1.0) * (sound["sampleRate"] / sound["sampleCount"]);

        fftArray += 1  # move everything up by one, hence no value less than 0.
        logFFT = (10 * np.log10(fftArray)) #Logarithmic
        scaleFFT = logFFT / 100  # Down scale between 0 and 1
        sound["frequencySpectrum"] = scaleFFT
        sound["spectrumLen"] = len(scaleFFT)



    # ############################################################################
    # ...
    # ############################################################################
    def convertUINT8audioToINT16(self, data):

        newData = (data.astype(np.int16) - math.floor(AudioHandler.UINTMAX / 2))
        return newData * math.floor(AudioHandler.SIXTEENBITMAX / (AudioHandler.UINTMAX / 2))


    # ############################################################################
    # Writes a sound to disk
    # ############################################################################
    def writeSoundToDir(self, sound, dir, name=None, sound2=None):

        try:
            os.mkdir(dir)
        except Exception as ex:
            pass

        if name is None:
            nextFullPathFileName = os.path.join(dir, 'outfile_'+str(random.randint(1000000000, 2000000000))+'.wav')
        else:
            nextFullPathFileName = os.path.join(dir, name+'.wav')

        scaledSoundL = self.scaleAudio(sound, 1/sound["scaling"])

        writableOutput = scaledSoundL["data"]

        if sound2 is not None:
            # Write a stereo sound!
            scaledSoundR = self.scaleAudio(sound2, 1 / sound2["scaling"])
            writableOutput = np.ones(dtype='int16', shape=(scaledSoundR["sampleCount"], 2))
            writableOutput[:,0]=scaledSoundL["data"]
            writableOutput[:,1]=scaledSoundR["data"]

        wavfile.write(nextFullPathFileName, scaledSoundL['sampleRate'], writableOutput)

    # ############################################################################
    # scale down audio from 16 bit int (-32768 - 32768) to float (0.0 - 1.0)
    # ############################################################################
    def downScaleAudio(self, upscaled):

        return ((upscaled * AudioHandler.K) / (2 * AudioHandler.SIXTEENBITMAX)) + AudioHandler.center

    # ############################################################################
    # scale up audio from float (0.0 - 1.0) to 16 bit int (-32768 - 32768)
    # ############################################################################
    def upScaleAudio(self, downscaled):

        return (((downscaled - AudioHandler.center) * 2 * AudioHandler.SIXTEENBITMAX) / AudioHandler.K).astype('int16')

    # ############################################################################
    # returns a piece of sound from a longer sound
    # ############################################################################
    def getAPieceOfSound(self, soundData, offset, length=None):

        if length is None:
            length = soundData['data'].shape[0]

        if length > soundData["scaledData"].shape[0] - offset:
                raise ValueError("defining to long slice of sound when picking a slice")

        return {
            "sampleRate": soundData["sampleRate"],
            "data": soundData["data"][offset:length+offset],
            "scaledData" : soundData["scaledData"][offset:length+offset],
            "frequencySpectrum": None,
            "spectrumLen": 0,
            "fileName": soundData["fileName"]+"Cut",
            "userDefinedName": soundData["userDefinedName"] + "/pick@"+str(offset)+":"+str(length),
            "sampleCount": length,
            "trackLengthSec": length/soundData["sampleRate"],
            "scaling": 1.0
        }

    # ############################################################################
    # Converts an output from inference into a sound dictionary
    # ############################################################################
    def createSoundFromInferenceOutput(self, sound, sampleRate=22100, name="fromInference", userDefinedName=""):

        return {
            "sampleRate": sampleRate,
            "data": self.upScaleAudio(sound),
            "scaledData" : sound,
            "frequencySpectrum": None,
            "spectrumLen": 0,
            "fileName": name,
            "userDefinedName": userDefinedName,
            "sampleCount": sound.shape[0],
            "trackLengthSec": sound.shape[0]/sampleRate,
            "scaling": 1.0
        }


    # ############################################################################
    # Working on raw data, mixing two sounds together by overlapping them.
    # A = Standard python array
    # B = typically numpy... or Standard python is fine..
    # ############################################################################
    def overlapRawData(self, soundA, soundB, overlap):

        if len(soundA) < overlap:  # Start of sound. Nothing to append to...
            for a in range(len(soundB)) :
                soundA.append(soundB[a])
            return soundA

        scalerStepSize = 1
        scalerA = overlap - (scalerStepSize / 2)
        scalerB = scalerStepSize / 2

        # Mix the two sounds together...
        for a in range(overlap):
            soundASample = soundA[-overlap + a]
            soundBSample = soundB[a]
            soundASample = soundASample * scalerA
            soundBSample = soundBSample * scalerB
            soundA[-overlap + a] = ((soundASample + soundBSample) / overlap).astype('float32')
            scalerA -= scalerStepSize
            scalerB += scalerStepSize

        # Append the rest..
        for a in range(len(soundB) - overlap):
            soundA.append(soundB[a + overlap])

        return soundA

    # ############################################################################
    # // SORRY! I should not need to do post processing, but in the stepping mode
    # I do this for now to be able to remove some white noise...
    # ############################################################################
    def lowPassFilter(self, sound, filterSteps):

        if filterSteps is 0:
            return sound

        sound['scaledData'] = np.convolve(sound['scaledData'], np.ones((filterSteps,)) / filterSteps, mode='same')

        # Convolution (sliding window avg) does not fill these first and last bytes...
        sound['scaledData'][0] = sound['scaledData'][1]
        sound['scaledData'][-1] = sound['scaledData'][-2]

        sound['data'] = self.upScaleAudio(sound['scaledData'])
        return sound


    # ############################################################################
    # Use this function to get e.g loss between two sounds..
    # If not equal length, only so many samples will be compared.
    # ############################################################################
    def soundDiff(self, soundA, soundB):

        len = 0
        if soundA["sampleCount"] < soundB["sampleCount"]:
            len = soundA["sampleCount"]
        else:
            len = soundB["sampleCount"]

        diff = soundA["data"][0:len] - soundB["data"][0:len]
        # Cast the diff-array to int64, as the summing up of this may overflow standard int32
        return np.sum(np.abs(diff.astype('int64'))) / len


    # ############################################################################
    # Returns a value based of the frequence differences in two sounds
    # ############################################################################
    def soundDiffFFT(self, soundA, soundB):

        fftArrayA = fft(soundA["data"])
        fftArrayB = fft(soundB["data"])
        fftArrayA = abs(fftArrayA)
        fftArrayB = abs(fftArrayB)

        diff = fftArrayA[0:soundA['sampleCount']] - fftArrayB[0:soundB['sampleCount']]

        return np.sum(np.abs(diff.astype('int64'))) / soundA['sampleCount']


    # ############################################################################
    # Calculate the average amplitude of a sound
    # ############################################################################
    def soundAvgAmplitude(self, sound):
        return np.sum(np.abs(sound["data"].astype('int64'))) / sound["sampleCount"]


    # ############################################################################
    # Adds or subtract two sounds and returns the result.
    # operation is either "+" or "-" // SORRY! Can be done with numpy operation. I know
    # Result is scaled to 50% to be in the proper operation range.
    # ############################################################################
    def addSubtractSounds(self, soundA, operation, soundB, scaling=0.5):

        if soundA["sampleCount"] != soundB["sampleCount"]:
            raise ValueError("Cannot do diff on sounds with different lengt. Use getAPieceOfSound() first")

        if operation == "-":
            result = (soundA["data"] - soundB["data"]).astype('int16')
        else:
            result = (soundA["data"] + soundB["data"]).astype('int16')

        result =  {
            "sampleRate": soundA["sampleRate"],
            "data": result.astype(np.int16),
            "scaledData" : None, # Cannot scale the audio until downscaled to proper levels below..
            "frequencySpectrum": None,
            "spectrumLen": 0,
            "fileName": "addedSounds"+soundA["fileName"]+soundB["fileName"],
            "userDefinedName": soundA["userDefinedName"]+"diff",
            "sampleCount": soundA["sampleCount"],
            "trackLengthSec": soundA["trackLengthSec"],
            "scaling": 1.0
        }

        return self.scaleAudio(result, scaling=scaling)

    # ############################################################################
    # Freely scales an audio volume up or down. The scaling factor is stored
    # in this sound so it is later possible to scale it back to proper volume
    # ############################################################################
    def scaleAudio(self, sound, scaling):

        result = (sound["data"] * scaling).astype('int16')

        return  {
            "sampleRate": sound["sampleRate"],
            "data": result,
            "scaledData" : self.downScaleAudio(result),
            "frequencySpectrum": None,
            "spectrumLen": 0,
            "fileName": sound["fileName"] + "scaled",
            "userDefinedName": sound["userDefinedName"]+"scaled",
            "sampleCount": sound["sampleCount"],
            "trackLengthSec": sound["trackLengthSec"],
            "scaling": scaling * sound["scaling"]
        }


# ############################################################################
# For testing purposes..
# ############################################################################
if __name__ == '__main__':
    a = AudioHandler()

    soundData = AudioHandler().readAllFilesInDir(defs.WAV_FILE_PATH)
    a.lowPassFilter(a.getAPieceOfSound(soundData[1], 0, 100), 3)

    exit(0)

    # Temp:
    slice = AudioHandler().getAPieceOfSound(soundData[0], random.randint(0,soundData[0]["sampleCount"] - 1 - 1000), 1000)
    AudioHandler().addFFTToSound(slice)

