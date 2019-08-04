from definitions import Definitions as defs

from sys import byteorder
from array import array
from struct import pack
import pyaudio
import wave
import threading
import os
import random
from time import sleep

# ############################################################################
# DEFINITIONS... Go ahead and tune..
# ############################################################################
CHUNK_SIZE = 1024  # bytes
RATE = 44100 # Hz
FORMAT = pyaudio.paInt16


# ############################################################################
# Global variables...
# ############################################################################
pause = True
abort = False
audioHandle = None

# ############################################################################
# Do the actual recording of sound...
# ############################################################################
def printDeviceInfo(pyAudio):
    try:
        info = pyAudio.get_default_input_device_info()

        if info['maxInputChannels'] < 2:
            print(f"Your default recording device {info['name']} seems to be mono!!??  Aborting")
            exit(-1)
        else:
            print(f"Using default stereo recording device: \"{info['name']}\"")
    except Exception as ex:
        print(f"Failed to check device... {ex}")

# ############################################################################
# Do the actual recording of sound...
# ############################################################################
def record():
    global pause
    global audioHandle

    stream = audioHandle.open(format=FORMAT, channels=2, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    r = array('h')

    while not pause:

        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

    sample_width = audioHandle.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    #audioHandle.terminate()

    return sample_width, r

# ############################################################################
# record and store recording to file...
# ############################################################################
def record_to_file(path):

    print(f" -- RECORDING -- \"{path}\" (Hit Enter to pause...)")
    sample_width, data = record()
    print(f"Saving \"{path}\" to disk")
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(2)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

# ############################################################################
# The recording thread waits for run or pause
# ############################################################################
def waitForRecordingCommand():

    global pause
    global abort
    defs.WAV_FILE_PATH


    while not abort:
        if pause:
            sleep(0.1)
        else:
            nextFileName = defs.WAV_FILE_PATH + '/recording_'+str(random.randint(1000000000, 2000000000))+'.wav'
            record_to_file(nextFileName)

    print("Recording thread exited")

# ############################################################################
# Wait for user input, e.g. Enter key..
# ############################################################################
def keyboarInput(thread):

    global pause
    global audioHandle
    global abort

    while True:
        sleep(0.5)
        inputChar = input('\n----------------------------\nHit Enter to start recording (q to quit)...')
        if inputChar == 'q':
            print("Exiting...")
            audioHandle.terminate()
            abort = True
            thread.join()
            print("Exiting program")
            exit(0)

        pause = False
        input()
        pause = True



# ############################################################################
# Main!
# ############################################################################
if __name__ == '__main__':

    if not os.path.exists(defs.WAV_FILE_PATH):
        os.makedirs(defs.WAV_FILE_PATH)

    audioHandle = pyaudio.PyAudio()
    printDeviceInfo(audioHandle)
    random.seed()
    t = threading.Thread(target=waitForRecordingCommand)
    t.start()

    keyboarInput(t)

