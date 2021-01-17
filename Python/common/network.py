import tensorflow as tf
import time
import numpy as np
import math

g_activation="lrelutanhexp"
g_CnnChannels = 24

#####################################################
# Define the CCN layers.
#####################################################
def defineNetworkLayers(layer, networkOutputLen):

    global g_CnnChannels

    inputLayer = tf.reshape(layer, [-1, int(layer.shape[1]), 1], name="reshapeInput")

    with tf.variable_scope(f"FClvl{'X'}_blck0") as scope:
        layer1CNN = tf.layers.conv1d(inputLayer, g_CnnChannels, 48, 2, padding='same', activation=myActivation, name="CNN1")
        layer2CNN = tf.layers.conv1d(layer1CNN, g_CnnChannels, 8, 2, padding='same', activation=myActivation, name="CNN2")
        layer3CNN = tf.layers.conv1d(layer2CNN, g_CnnChannels, 6, 2, padding='same', activation=myActivation, name="CNN3")
        layer4CNN = tf.layers.conv1d(layer3CNN, g_CnnChannels, 5, 2, padding='same', activation=myActivation, name="CNN4")
        layer5CNN = tf.layers.conv1d(layer4CNN, g_CnnChannels, 4, 2, padding='same', activation=myActivation, name="CNN5")
        layer6CNN = tf.layers.conv1d(layer5CNN, g_CnnChannels, 3, 2, padding='same', activation=myActivation, name="CNN6")
        layer7CNN = tf.layers.conv1d(layer6CNN, g_CnnChannels, 2, 2, padding='same', activation=myActivation, name="CNN7")
        layer8CNN = tf.layers.conv1d(layer7CNN, g_CnnChannels, 2, 2, padding='same', activation=myActivation, name="CNN8")
        layer8Flat = tf.reshape(layer8CNN, [-1, int(layer8CNN.shape[1] * layer8CNN.shape[2])], name="reshape9")

    layer9FC = createFCBlock(layer8Flat, int(int(layer8CNN.shape[1] * layer8CNN.shape[2]) / 4), 2, 9, myActivation)
    layer9Flat = tf.reshape(layer9FC, [-1, int(int(layer8CNN.shape[1])/2) * int(layer8CNN.shape[2]), 1])

    with tf.variable_scope(f"FClvl{'Y'}_blck0") as scope:
        layer10CNN = tf.layers.conv1d(layer9Flat, g_CnnChannels, 18, 2, padding='same', activation=myActivation, name="CNN10")
        layer11CNN = tf.layers.conv1d(layer10CNN, g_CnnChannels, 3, 2, padding='same', activation=myActivation, name="CNN11")
        layer11Flat = tf.reshape(layer11CNN, [-1, int(layer11CNN.shape[1] * layer11CNN.shape[2])], name="reshape11")

    layer12FC = createFCBlock(layer11Flat, int(networkOutputLen / 2), 2, 11, tf.nn.tanh)
    return layer12FC

def createFCBlock(input, blockSize, blocks, level, activation):

    blockList = []

    for block in range(blocks):
        with tf.variable_scope(f"FClvl{level}_blck{block}") as scope:
            blockList.append(tf.contrib.layers.fully_connected(input, blockSize, activation_fn=None))

    return activation(tf.concat(blockList, 1, name=f"FCBlckConcatLvl{level}"))

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
            retValy_modelFC = defineNetworkLayers(retValxFC, networkOutputLen)

        return retValgraphFC, retValsessionFC, retValxFC, retValy_modelFC

#####################################################
# Calculates the output from the FC network
#####################################################
def myActivation(layer, activationAlpha=0.02, dropoutRate=0.1):

    global g_activation
    if g_activation is "tanh":
        layer = tf.nn.tanh(layer)
    elif g_activation is "leakytanh":
        layer = tf.nn.tanh(layer) + 0.05 * layer
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
    elif g_activation is "lrelutanh":
        layer = tf.nn.leaky_relu(layer) * tf.nn.tanh(layer)
    elif g_activation is "relutanhexp":
        layer = tf.nn.relu(layer) * tf.nn.tanh(layer) * layer
    elif g_activation is "lrelutanhexp":
        layer = tf.nn.leaky_relu(layer) * tf.nn.tanh(layer) * layer
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
def restoreGraphFromDisk(sessionFC, graphFC, fullGraphPath, forceClean = False):
    with graphFC.as_default() as g:

        try:

            if forceClean:
                print("Not trying to read graph from disk... ")
                raise
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
