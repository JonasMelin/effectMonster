import tensorflow as tf
import time
import numpy as np
import math

#####################################################
# Define the neural network.
#####################################################
def defineFCModel(networkInputLen, networkOutputLen, per_process_gpu_memory_fraction=0.85):

    retValgraphFC = tf.Graph()
    retValsessionFC = tf.Session(graph=retValgraphFC, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)))

    with retValgraphFC.as_default() as g:

        # Input!
        retValxFC = tf.placeholder(tf.float32, shape=[None, networkInputLen], name='xConv')

        # CNN feature extraction layers
        layer = tf.reshape(retValxFC, [-1, int(retValxFC.shape[1]), 1])
        layer = tf.layers.conv1d(layer, 32, 48, 2, padding='same', activation=tf.nn.leaky_relu)
        layer = tf.layers.conv1d(layer, 24, 32, 2, padding='same', activation=tf.nn.leaky_relu)
        layer = tf.layers.conv1d(layer, 18, 24, 2, padding='same', activation=tf.nn.leaky_relu)
        layer = tf.layers.conv1d(layer, 12, 18, 2, padding='same', activation=tf.nn.leaky_relu)

        layer = tf.reshape(layer, [-1, int(layer.shape[1] * layer.shape[2])])

        # Fully connected layers

        layer = tf.contrib.layers.fully_connected(layer, int(int(layer.shape[1]) * 0.5), activation_fn=tf.nn.leaky_relu)
        retValy_modelFC = tf.contrib.layers.fully_connected(layer, networkOutputLen, activation_fn=tf.keras.activations.tanh)

        return retValgraphFC, retValsessionFC, retValxFC, retValy_modelFC

#####################################################
# Calculates the output from the FC network
#####################################################
def getFCOutput(dataX, sessionFC, graphFC, xFC, y_modelFC):

    feed_dict = {xFC: dataX}

    with graphFC.as_default() as g:
        return sessionFC.run(y_modelFC, feed_dict=feed_dict)

#####################################################
# Restores the graph from disk. If not exist,
# initializes all global variables.
#####################################################
def restoreGraphFromDisk(sessionFC, graphFC, fullGraphPath):
    with graphFC.as_default() as g:
        try:
            tf.train.Saver().restore(sessionFC, fullGraphPath)
            print(f"Successfully restored variables from disk! {fullGraphPath}")
        except:
            print(f"Failed to restore variables from disk! {fullGraphPath}")
            sessionFC.run(tf.global_variables_initializer())

#####################################################
# Runs inference with sliding window over an entire sound.
#####################################################
def runInferenceOnSoundSampleBySample(soundData, audio, networkInputLen,
                                      networkOutputLen, inferenceOverlap, effectiveInferenceOutputLen,
                                      BATCH_SIZE_INFERENCE_FULL_SOUND, sessionFC, graphFC, xFC, y_modelFC):

    inferenceCounter = 0
    writeCounter = networkInputLen - effectiveInferenceOutputLen
    outRawData = np.zeros(soundData["sampleCount"])
    outRawData[:] = audio.center
    done = False

    assert networkOutputLen > inferenceOverlap
    assert networkOutputLen >= effectiveInferenceOutputLen

    infTimeOnlyTot = 0

    try:
        while not done:
            inputBatch = []

            for e in range(math.floor(BATCH_SIZE_INFERENCE_FULL_SOUND / effectiveInferenceOutputLen)):
                nextDataSlize = audio.getAPieceOfSound(soundData, inferenceCounter, networkInputLen)
                reshapedDataSlize = nextDataSlize["scaledData"].reshape(networkInputLen)
                inputBatch.append(reshapedDataSlize)
                inferenceCounter += effectiveInferenceOutputLen - inferenceOverlap
                if inferenceCounter >= (soundData["sampleCount"] - networkInputLen - 1):
                    done = True
                    break

            st = time.time()
            arrayOfQuantizedsamples = getFCOutput(inputBatch, sessionFC, graphFC, xFC, y_modelFC)
            infTimeOnlyTot += time.time() - st

            outputConvertedBatch = []
            for nextQSample in arrayOfQuantizedsamples:

                if inferenceOverlap is 0:
                    # No overlap... Just copy the data from inference
                    for i in range((networkOutputLen - effectiveInferenceOutputLen)):
                        outputConvertedBatch.append(nextQSample[i + networkOutputLen - effectiveInferenceOutputLen])
                else:
                    # Overlap!
                    outputConvertedBatch = audio.overlapRawData(outputConvertedBatch, nextQSample, inferenceOverlap)

            outRawData[writeCounter:writeCounter + len(outputConvertedBatch)] = outputConvertedBatch
            writeCounter += len(outputConvertedBatch) - inferenceOverlap

    except Exception as ex:
        pass

    soundOutput = audio.createSoundFromInferenceOutput(outRawData, sampleRate=soundData["sampleRate"])
    #lowPassFiltered = self.audio.lowPassFilter(soundOutput, self.params['lowPassFilterSteps'])

    return soundOutput, infTimeOnlyTot / soundData['trackLengthSec']
