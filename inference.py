import audioHandler
import network
from definitions import Definitions as defs
import audioPlotter
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # Disable GPU..
import tensorflow as tf

networkInputLen = 1024
networkOutputLen = 128
inferenceOverlap = 8
fullGraphPath = os.path.join(defs.GRAPH_PATH, 'latest')

class inference:
    def __init__(self):
        self.audio = audioHandler.AudioHandler()

        self.graphFC, self.sessionFC, self.xFC, self.y_modelFC = network.defineFCModel(
            networkInputLen, networkOutputLen)
        network.restoreGraphFromDisk(self.sessionFC, self.graphFC, fullGraphPath)


    def run(self):
        soundDataList = self.audio.readAllFilesInDir(defs.WAV_FILE_PATH)

        if soundDataList[0]["fileName"] == soundDataList[1]["fileName"]:
            # In this case, one stereo file was provided, use one channel as input, the other as label
            inputSoundRaw = soundDataList[1]
            labelSoundRaw = soundDataList[0]
        else:
            # Here, two mono files are provided and (hopefully) named label and input. Use the files as such
            for z in range(2):
                if "input" in soundDataList[z]["fileName"]:
                    inputSoundRaw = soundDataList[z]

        if inputSoundRaw is None:
            print("Provide either one mono file named input, or one stereo track with input data (L/R shall be identical!)")
            exit(90)

        print("Running inference... Please wait...")
        infOutSound, infTime = network.runInferenceOnSoundSampleBySample(inputSoundRaw, self.audio,
                                                                         networkInputLen,
                                                                         networkOutputLen,
                                                                         inferenceOverlap,
                                                                         networkOutputLen,
                                                                         networkOutputLen,
                                                                         self.sessionFC, self.graphFC, self.xFC,
                                                                         self.y_modelFC)

        print(
            f"Complete! Writing sound to {defs.WAV_FILE_OUTPUT}. Inference using CPU only took {infTime:.2f} seconds/seconds")
        
        self.audio.writeSoundToDir(infOutSound, defs.WAV_FILE_OUTPUT, "EffectOut_" + inputSoundRaw["fileName"])


if __name__ == "__main__":

    inference().run()
