import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # Disable GPU..

import audioHandler
import network
from definitions import Definitions as defs


batchSize = 128
inferenceOverlap = 80

calulationUnit="CPU"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    # Running the GPU. This means run @full performance
    batchSize = 20000
    calulationUnit="GPU"

class inference:
    def __init__(self):
        self.audio = audioHandler.AudioHandler()

        self.graphFC, self.sessionFC, self.xFC, self.y_modelFC = network.defineFCModel(
            defs.networkInputLen, defs.networkOutputLen, per_process_gpu_memory_fraction=0.20)
        network.restoreGraphFromDisk(self.sessionFC, self.graphFC, defs.fullGraphPath)


    def run(self):
        soundDataList = self.audio.readAllFilesInDir(defs.WAV_FILE_PATH)

        if len(soundDataList) > 1 and (soundDataList[0]["fileName"] == soundDataList[1]["fileName"]):
            # In this case, one stereo file was provided, use one channel as input, the other as label
            inputSoundRaw = soundDataList[1]
        else:
            # Here, two mono files are provided and (hopefully) named label and input. Use the files as such
            for z in range(len(soundDataList)):
                if "input" in soundDataList[z]["fileName"]:
                    inputSoundRaw = soundDataList[z]

        if inputSoundRaw is None:
            print("Provide either one mono file named input, or one stereo track with input data (L/R shall be identical!)")
            exit(90)

        print(f"Using {calulationUnit} to run inference for file {inputSoundRaw['fileName']}... Please wait...")
        infOutSound, infTime = network.runInferenceOnSoundSampleBySample(inputSoundRaw, self.audio,
                                                                         defs.networkInputLen, defs.networkOutputLen,
                                                                         inferenceOverlap, defs.networkOutputLen,
                                                                         batchSize, self.sessionFC, self.graphFC, self.xFC,
                                                                         self.y_modelFC)

        print(f"Complete! Writing sound to {defs.WAV_FILE_OUTPUT}. Inference using {calulationUnit} took {infTime:.4f} seconds/seconds")
        self.audio.writeSoundToDir(infOutSound, defs.WAV_FILE_OUTPUT, "EffectOut_" + inputSoundRaw["fileName"])

if __name__ == "__main__":

    inference().run()
