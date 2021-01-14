import tensorflow as tf
import time
import numpy as np
import math

g_activation="relutanh"

#####################################################
# Define the CCN layers.
#####################################################
def defineCNNLayers(layer):
    layer = tf.reshape(layer, [-1, int(layer.shape[1]), 1])
    layer = tf.layers.conv1d(layer, 64, 6, 2, padding='same', activation=myActivation)
    layer = tf.layers.conv1d(layer, 64, 5, 2, padding='same', activation=myActivation)
    layer = tf.layers.conv1d(layer, 64, 5, 2, padding='same', activation=myActivation)
    layer = tf.layers.conv1d(layer, 64, 5, 2, padding='same', activation=myActivation)
    layer = tf.layers.conv1d(layer, 64, 4, 2, padding='same', activation=myActivation)
    #layer = tf.layers.conv1d(layer, 64, 3, 2, padding='same', activation=myActivation)
    #layer = tf.layers.conv1d(layer, 64, 2, 2, padding='same', activation=myActivation)

    layer = tf.reshape(layer, [-1, int(layer.shape[1] * layer.shape[2])])
    return layer

#####################################################
# Define the neural network.
#####################################################
def defineFCModel(networkInputLen, networkOutputLen, per_process_gpu_memory_fraction=0.85, activation="tanh"):

    global g_activation
    g_activation = activation

    retValgraphFC = tf.Graph()
    retValsessionFC = tf.Session(graph=retValgraphFC, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)))

    with retValgraphFC.as_default() as g:

        # Input!
        retValxFC = tf.placeholder(tf.float32, shape=[None, networkInputLen], name='xConv')

        with tf.variable_scope('Variables') as scope:
            layerCNN1 = defineCNNLayers(retValxFC)
            layer = layerCNN1 #tf.concat([layerFC1], 1)
            layer = myActivation(tf.contrib.layers.fully_connected(layer, networkOutputLen, activation_fn=None))
            retValy_modelFC = tf.contrib.layers.fully_connected(layer, networkOutputLen, activation_fn=tf.keras.activations.tanh)

        return retValgraphFC, retValsessionFC, retValxFC, retValy_modelFC

#####################################################
# Calculates the output from the FC network
#####################################################
def myActivation(layer, activationAlpha=0.02, dropoutRate=0.1):

    global g_activation
    if g_activation is "tanh":
        layer = tf.nn.tanh(layer)
    elif g_activation is "swish":
        layer = tf.nn.swish(layer)
    elif g_activation is "lrelu":
        layer = tf.nn.leaky_relu(layer)
    elif g_activation is "relu":
        layer = tf.nn.relu(layer)
    elif g_activation is "relusigmoid":
        layer = tf.nn.relu(layer) * tf.nn.sigmoid(layer)
    elif g_activation is "relutanh":
        layer = tf.nn.relu(layer) * tf.nn.tanh(layer)
    else:
        raise ValueError(f"BAD ACTIVATION {g_activation}")

    return layer

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
