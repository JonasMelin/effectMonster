import audioPlotter
import audioHandler
from definitions import Definitions as defs
import json
import os
import random
import math
import numpy as np
import time
import threading
from threading import Lock

# test commit to understand github contributors. 

# ############################################################################
# Main class for running all training.
# ############################################################################
class MainTrainer:

    # ############################################################################
    # Construct
    # ############################################################################
    def __init__(
                self,
                disable_gpu = False,
                per_process_gpu_memory_fraction = 0.85,
                USE_RELU = True,  # False => sigmoid
                BATCH_SIZE = 750,  # Batch size for training.
                BATCH_INF_SIZE = 1000,  # When running inference for stats purpose, how large batch to the GPU
                BATCH_SIZE_INFERENCE_FULL_SOUND = 10000,  # Batch size When running inference to generate full audio file
                STATS_EVERY = 250,  # How often (skipping steps) to run inference to gather stats.
                validationPercent = 0.06,  # e.g. 0.1 means 10% of the length of total sound will be validation
                MIN_STEPS_BETWEEN_SAVES = 5000,
                stride1 = 1,
                filterSize1 = 32,
                numberFilters1 = 4,
                stride2 = 2,
                filterSize2 = 64,
                numberFilters2 = 6,
                stride3 = 2,
                filterSize3 = 5,
                numberFilters3 = 18,
                stride4=2,
                filterSize4=5,
                numberFilters4=48,
                stride5=1,
                filterSize5=5,
                numberFilters5=12,
                hidden_layers = 8,
                learning_rate = 0.00005,
                learning_rate_decay = 1000000 , # Higher gives slower decay
                networkInputLen = 1024,
                graphName = 'latest',
                maxTrainingSamplesInMem=250000,
                uniqueSessionNumber = str(random.randint(10000000, 99000000))):

        global tf

        if disable_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"

        import tensorflow as tf

        self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction

        self.params = {
            'learning_rate_decay' : math.floor(learning_rate_decay / BATCH_SIZE),
            'USE_RELU' : USE_RELU,
            'BATCH_SIZE' : BATCH_SIZE,  # Batch size for training.
            'BATCH_INF_SIZE' : BATCH_INF_SIZE,  # When running inference for stats purpose, how large batch to the GPU
            'BATCH_SIZE_INFERENCE_FULL_SOUND' : BATCH_SIZE_INFERENCE_FULL_SOUND,  # Batch size When running inference to generate full audio file
            'STATS_EVERY' : STATS_EVERY,  # How often (skipping steps) to run inference to gather stats.
            'validationPercent' : validationPercent,  # e.g. 0.1 means 10% of the length of maxTrainSamples will be validation
            'MIN_STEPS_BETWEEN_SAVES' : MIN_STEPS_BETWEEN_SAVES,
            'stride1': stride1,
            'filterSize1' : filterSize1,
            'numberFilters1' : numberFilters1,
            'stride2' : stride2,
            'filterSize2' : filterSize2,
            'numberFilters2' : numberFilters2,
            'stride3' : stride3,
            'filterSize3' : filterSize3,
            'numberFilters3' : numberFilters3,
            'stride4': stride4,
            'filterSize4': filterSize4,
            'numberFilters4': numberFilters4,
            'stride5': stride5,
            'filterSize5': filterSize5,
            'numberFilters5': numberFilters5,
            'hidden_layers' : hidden_layers,
            'learning_rate' : learning_rate,
            'networkInputLen' : networkInputLen,
            'graphName': graphName,
            'uniqueSessionNumber' : uniqueSessionNumber,
            'maxTrainingSamplesInMem': maxTrainingSamplesInMem
        }

        print(self.params)

        self.tensorboardFullPath = os.path.join(defs.TENSORBOARD_PATH, self.params['uniqueSessionNumber'])
        self.fullGraphPath = os.path.join(defs.GRAPH_PATH, self.params['graphName'])
        self.defineFCModel()
        self.audio = audioHandler.AudioHandler()
        self.plotter = audioPlotter.AudioPlotter()
        self.trainingData = []
        self.validationData = []
        self.globalLock = Lock()
        self.blockNextImgPrintOut = False
        self.slowMode = False

        try:
            os.mkdir(self.tensorboardFullPath)
        except Exception as ex:
            pass
        try:
            os.mkdir(defs.METADATA_OUTPUT)
        except Exception as ex:
            pass

        metadataFilePath  = os.path.join(defs.METADATA_OUTPUT, str(self.params["uniqueSessionNumber"]))
        with open(metadataFilePath + ".json", 'w') as fp:
            json.dump(self.params, fp)

        threading.Thread(target=self.keyboardReader).start()

    def keyboardReader(self):
        print("Press b to block next picture print out...")
        while True:

            val = input()
            if val is "b":
                self.blockNextImgPrintOut = True
                print("Blocking next image print out")
            if val is "s":
                self.slowMode  = not self.slowMode
                if self.slowMode:
                    print("SLOW MODE!! TRAINING SLOW TO ALLOW OTHER PROGRAMS TO USE GPU")
                else:
                    print("FULL MODE!! TRAINING FAST. Might make other things run slower...")


    #####################################################
    # Define the neural network.
    #####################################################
    def defineFCModel(self):

        self.graphFC = tf.Graph()
        self.sessionFC = tf.Session(graph=self.graphFC, config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction)))

        with self.graphFC.as_default() as g:

            # Input!
            self.xFC = tf.placeholder(tf.float32, shape=[None, self.params['networkInputLen']], name='xConv')

            # CNN 1d 2 layers, 1 fully connected layer
            layerCNN, filterCount, aggregatedStride = self.newConvLayer(self.xFC, self.params['stride1'], 1, self.params['filterSize1'], self.params['numberFilters1'], self.params['USE_RELU'])
            print(f"OutputSize after 1st CNN: {math.floor((filterCount) * self.params['networkInputLen']  / aggregatedStride)}")
            layerCNN, filterCount, aggregatedStride = self.newConvLayer(layerCNN, self.params['stride2'], self.params['numberFilters1'], self.params['filterSize2'], self.params['numberFilters2'], self.params['USE_RELU'], aggregatedStride)
            print(f"OutputSize after 2nd CNN: {math.floor((filterCount) * self.params['networkInputLen']  / aggregatedStride)}")
            #layerCNN, filterCount, aggregatedStride = self.newConvLayer(layerCNN, self.params['stride3'], self.params['numberFilters2'], self.params['filterSize3'], self.params['numberFilters3'], self.params['USE_RELU'], aggregatedStride)
            #print(f"OutputSize after 3rd CNN: {math.floor((filterCount) * self.params['networkInputLen']  / aggregatedStride)}")
            #layerCNN, filterCount, aggregatedStride = self.newConvLayer(layerCNN, self.params['stride4'], self.params['numberFilters3'], self.params['filterSize4'], self.params['numberFilters4'], self.params['USE_RELU'], aggregatedStride)
            #print(f"OutputSize after 4th CNN: {math.floor((filterCount) * self.params['networkInputLen'] / aggregatedStride)}")
            #layerCNN, filterCount, aggregatedStride = self.newConvLayer(layerCNN, self.params['stride5'], self.params['numberFilters4'], self.params['filterSize5'], self.params['numberFilters5'], self.params['USE_RELU'], aggregatedStride)
            #print(f"OutputSize after 5th CNN: {math.floor((filterCount) * self.params['networkInputLen'] / aggregatedStride)}")


            #Flatten before pushing into fully connected
            layerCNN = tf.reshape(layerCNN, [-1, math.floor((filterCount) * self.params['networkInputLen']  / aggregatedStride)])

            # matmul the CNN output with the input! skip-through!
            #layerCNN = self.new_fc_layer(layerCNN, int(layerCNN.shape[1]), self.params['networkInputLen'],  self.params['USE_RELU'])

            #layerFC = self.new_fc_layer(self.xFC, int(self.xFC.shape[1]), math.floor(int(self.xFC.shape[1]) * 1.0), self.params['USE_RELU'])
            #layerFC = self.new_fc_layer(layerFC, int(layerFC.shape[1]), self.params['networkInputLen'], self.params['USE_RELU'])

            #w1 = self.new_weights(shape=[self.params['networkInputLen'], self.params['networkInputLen']])
            #w2 = self.new_weights(shape=[self.params['networkInputLen'], self.params['networkInputLen']])
            #layer = tf.nn.sigmoid(tf.matmul(layerCNN, w1) + tf.matmul(layerFC, w2))

            layer = self.new_fc_layer(layerCNN, int(layerCNN.shape[1]), math.floor(int(layerCNN.shape[1]) * 1.0), self.params['USE_RELU'])
            print(f"OutputSize after first FC layer: {int(layer.shape[1])}")


            # Extra fully connected
            for a in range(self.params['hidden_layers']):
                layer = self.new_fc_layer(layer, int(layer.shape[1]), math.floor(int(layer.shape[1]) * 0.45), self.params['USE_RELU'])
                print(f"OutputSize after next FC layer: {int(layer.shape[1])}")

            self.y_modelFC = self.new_fc_layer(layer,  int(layer.shape[1]), defs.quantSteps, False)
            print(f"Created {self.params['hidden_layers']} Fully connected layers...")

            # cost functions an optimizers..
            self.y_true_FC = tf.placeholder(tf.float32, shape=[None, defs.quantSteps], name='y_trueFC')
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.params['learning_rate'], global_step, self.params['learning_rate_decay'], 0.99, staircase=True)

            #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_modelFC, labels=self.y_true_FC))
            #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #self.optimizerFC = optimizer.minimize(cost)

            cost = tf.reduce_mean(tf.square(self.y_modelFC - self.y_true_FC))
            self.optimizerFC = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

            # Tensorboard
            tf.summary.scalar("1_loss", cost)
            tf.summary.scalar("3_learning_rate", learning_rate)
            self.merged_summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.tensorboardFullPath)


            try:
                tf.train.Saver().restore(self.sessionFC, self.fullGraphPath)
                print(f"Successfully restored variables from disk! {self.fullGraphPath}")
            except:
                print(f"Failed to restore variables from disk! {self.fullGraphPath}")
                self.sessionFC.run(tf.global_variables_initializer())



    #####################################################
    # Runs inference with sliding window over an entire sound.
    # This function is for the step-one-sample algorithm
    #####################################################
    def runInferenceOnSoundSampleBySample(self, soundData):

        inferenceCounter = 0
        writeCounter = self.params['networkInputLen'] - 1
        outRawData = np.zeros(soundData["sampleCount"])
        outRawData[:] = 0.5
        done = False

        try:
            while not done:
                inputBatch = []

                for e in range(self.params['BATCH_SIZE_INFERENCE_FULL_SOUND']):
                    nextDataSlize = self.audio.getAPieceOfSound(soundData, inferenceCounter, self.params['networkInputLen'])
                    reshapedDataSlize = nextDataSlize["scaledData"].reshape(self.params['networkInputLen'])
                    inputBatch.append(reshapedDataSlize)
                    inferenceCounter += 1
                    if inferenceCounter >= (soundData["sampleCount"] - self.params['networkInputLen'] - 1):
                        done = True
                        break

                arrayOfQuantizedsamples = self.getFCOutput(inputBatch)

                outputConvertedBatch = []
                for nextQSample in arrayOfQuantizedsamples:
                    if defs.USE_QUANT_OUT:
                        outputConvertedBatch.append(self.audio.quantArrayToSample(nextQSample))
                    else:
                        outputConvertedBatch.append(nextQSample) ## ToDo: Is this really ok??

                outRawData[writeCounter:writeCounter + len(outputConvertedBatch)] = outputConvertedBatch
                writeCounter += self.params['BATCH_SIZE_INFERENCE_FULL_SOUND']


        except Exception as ex:
            pass
        return self.audio.createSoundFromInferenceOutput(outRawData, sampleRate=soundData["sampleRate"])


    #####################################################
    # Load all training data into RAM, but keep training
    # during this process...
    #
    #####################################################
    def threadFuncPrepareData(self, inputSound, labelSound, dataOutput, name="", continouslyReplaceWithNewData=False):

        sampleSteps = inputSound["sampleCount"] - 1 - self.params['networkInputLen']

        if continouslyReplaceWithNewData:
            loopCount = self.params['maxTrainingSamplesInMem']
        else:
            loopCount = sampleSteps

        while(True):

            for r in range(loopCount):

                if not continouslyReplaceWithNewData:
                    # Take all provided samples from inputSound and prepare them
                    nextInputDataSlize = self.audio.getAPieceOfSound(inputSound, r, self.params['networkInputLen'])
                    nextLabelDataSlize = self.audio.getAPieceOfSound(labelSound, r, self.params['networkInputLen'])
                else:
                    # Take a random set of samples
                    sampleStartPos = random.randint(0,sampleSteps)
                    nextInputDataSlize = self.audio.getAPieceOfSound(inputSound, sampleStartPos, self.params['networkInputLen'])
                    nextLabelDataSlize = self.audio.getAPieceOfSound(labelSound, sampleStartPos, self.params['networkInputLen'])

                input = nextInputDataSlize["scaledData"]
                label = np.array(nextLabelDataSlize["scaledData"][defs.sampleToPredict]).reshape(1)

                if defs.USE_QUANT_OUT:
                    quantArrayLabel = self.audio.sampleToQuantArray(label[0])
                else:
                    quantArrayLabel = label

                reshapedInput = input.reshape(-1, self.params['networkInputLen'])
                reshapedLabel = label.reshape(-1, 1)

                if defs.USE_QUANT_OUT:
                    reshapedQuantArrayLabel = quantArrayLabel.reshape(-1, defs.quantSteps)
                else:
                    reshapedQuantArrayLabel = None

                # Python is not really multi threadded! So, yield for the main thread..
                if r % 50 == 0:
                    time.sleep(0.001)
                time.sleep(0)

                with self.globalLock:
                    finalData = {'input': reshapedInput, 'label': reshapedLabel, 'quantArrayLabel': reshapedQuantArrayLabel}

                    if len(dataOutput) < loopCount:
                        dataOutput.append(finalData)
                    else:
                        randomReplacePos = random.randint(0, loopCount - 1)
                        dataOutput[randomReplacePos] = finalData

            if not continouslyReplaceWithNewData:
                print(f"{name} data preparation completed. Everything loaded into RAM! {r} examples loaded")
                break
            else:
                print("Added a batch with new fresh training samples.. Taking a tiny break and letting the network train..")
                time.sleep(30) # Take a little break until we continue to replace the training data with new samples.

    #####################################################
    # Calculates a score that means the higher the better
    # and should "always" be comparable... You need variance
    # similar to label variance in order for a sound to be
    # good, in combinatino with a low error value.
    #####################################################
    def calcSuperScore(self, labelVar, infVar, error, step):

        if step < 5:
            return 0.0

        if infVar < 0.00000001:
            infVar  = 0.00000001

        retVal = 0.0

        try:
            varDiff = math.fabs(labelVar / infVar)  # Should be = 1 when perfect
            errSqTimesDiff = error ** 2 * varDiff
            if (errSqTimesDiff) < 0.00000001:
                errSqTimesDiff = 0.00000001
            retVal = 1 / errSqTimesDiff
        except Exception as ex:
            pass

        return retVal


    #####################################################
    # Train a network
    #####################################################
    def trainToReproduceSoundOneValueAtTheTime(self, slicesPerLoop = 250):

        trainTimePerSample_us = 0
        maxSuperScore = -999999.9
        soundWriteCounter = -10000000
        superScoreList = np.zeros(shape=10)
        superScoreCount = 0
        inputSoundRaw = None
        labelSoundRaw = None

        soundDataList = self.audio.readAllFilesInDir(defs.WAV_FILE_PATH)

        if len(soundDataList) != 2:
            print(f"Did not find one stereo track or two mono tracks.. {defs.WAV_FILE_PATH}. Exiting..")
            exit(89)

        if soundDataList[0]["fileName"] == soundDataList[1]["fileName"]:
            # In this case, one stereo file was provided, use one channel as input, the other as label
            inputSoundRaw = soundDataList[1]
            labelSoundRaw = soundDataList[0]
        else:
            # Here, two mono files are provided and (hopefully) named label and input. Use the files as such
            for z in range(2):
                if "label" in soundDataList[z]["fileName"]:
                    labelSoundRaw = soundDataList[z]
                if "input" in soundDataList[z]["fileName"]:
                    inputSoundRaw = soundDataList[z]

        if labelSoundRaw is None or inputSoundRaw is None:
            print("Provide either two mono files named label and input, or one stereo track with input and label as left anf right channel")
            exit(90)

        validationSampleCount = math.floor(self.params['validationPercent'] * inputSoundRaw['sampleCount'])
        validationStartPoint = inputSoundRaw['sampleCount'] - validationSampleCount

        inputSound = self.audio.getAPieceOfSound(inputSoundRaw, 0, validationStartPoint - 1)
        labelSound = self.audio.getAPieceOfSound(labelSoundRaw, 0, validationStartPoint - 1)
        inputSoundVal = self.audio.getAPieceOfSound(inputSoundRaw, validationStartPoint, validationSampleCount)
        labelSoundVal = self.audio.getAPieceOfSound(labelSoundRaw, validationStartPoint, validationSampleCount)

        threading.Thread(target=self.threadFuncPrepareData, args=(inputSound,labelSound, self.trainingData, "training", True)).start()
        threading.Thread(target=self.threadFuncPrepareData, args=(inputSoundVal, labelSoundVal, self.validationData, "validation", False)).start()

        while len(self.trainingData) < self.params['BATCH_SIZE'] or \
                len(self.trainingData) < self.params['BATCH_INF_SIZE'] or \
                len(self.validationData) < self.params['BATCH_INF_SIZE']:
            print("Waiting for data to be processed")
            time.sleep(1)

        self.audio.writeSoundToDir(inputSound, defs.WAV_FILE_OUTPUT, "InputSoundTrain")
        self.audio.writeSoundToDir(labelSound, defs.WAV_FILE_OUTPUT, "LabelSoundTrain")
        self.audio.writeSoundToDir(inputSoundVal, defs.WAV_FILE_OUTPUT, "InputSoundValidation")
        self.audio.writeSoundToDir(labelSoundVal, defs.WAV_FILE_OUTPUT, "LabelSoundValidation")
        #self.plotter.plotSoundSimple(labelSound, None, None, None, False, True)

        for r in range(11111111):

            if r % math.floor(self.params['STATS_EVERY']) is 0:

                #
                # GREATE METRICS TO TENSORBOARD
                #
                error = 0
                rawQuantError = 0
                errorVariance = 0
                infQuantOutVariance = 0
                startTime = time.time()


                with self.globalLock:
                    inputBatch = np.zeros(shape=(self.params['BATCH_INF_SIZE'], self.params['networkInputLen']))
                    labelBatch = np.zeros(shape=(self.params['BATCH_INF_SIZE'], defs.quantSteps))
                    finalOutInfVariance = np.zeros(shape=(self.params['BATCH_INF_SIZE']))
                    finalOutLabelVariance = np.zeros(shape=(self.params['BATCH_INF_SIZE']))

                    for p in range(self.params['BATCH_INF_SIZE']):
                        randomIndex = random.randint(0, len(self.validationData) - 1)
                        inputBatch[p] = self.validationData[randomIndex]["input"]
                        if defs.USE_QUANT_OUT:
                            labelBatch[p] = self.validationData[randomIndex]["quantArrayLabel"]
                        else:
                            labelBatch[p] = self.validationData[randomIndex]["label"]

                    outputArray = self.getFCOutput(inputBatch)

                    for index, nextOutput in enumerate(outputArray):
                        if defs.USE_QUANT_OUT:
                            infOutput = self.audio.quantArrayToSample(nextOutput)
                            labelOutput = self.audio.quantArrayToSample(labelBatch[index])
                        else:
                            infOutput = nextOutput[0]
                            labelOutput = labelBatch[index][0]

                        finalOutInfVariance[index] = infOutput
                        finalOutLabelVariance[index] = labelOutput
                        lastErr = math.fabs(infOutput - labelOutput)
                        error += lastErr

                        quantErrorArray = np.abs(nextOutput - labelBatch[index])
                        errorVariance += np.var(quantErrorArray)
                        rawQuantError += np.average(quantErrorArray)
                        infQuantOutVariance += np.var(nextOutput)

                infTime = time.time() - startTime
                inferenceTime_ms = 1000 * infTime
                inferenceTimePerMainLoop_ms = inferenceTime_ms / self.params['STATS_EVERY']
                trainTimePerMainLoop_s = trainTimePerSample_us * self.params['BATCH_SIZE'] / 1000000
                error /= self.params['BATCH_INF_SIZE']
                errorVariance /= self.params['BATCH_INF_SIZE']
                rawQuantError /= self.params['BATCH_INF_SIZE']
                infQuantOutVariance /= self.params['BATCH_INF_SIZE']
                infFinalOutVariance = np.var(finalOutInfVariance)
                labelFinalOutVariance = np.var(finalOutLabelVariance)
                finalOutVarianceQuota = infFinalOutVariance / labelFinalOutVariance

                superScoreList[superScoreCount % len(superScoreList)] = self.calcSuperScore(labelFinalOutVariance, infFinalOutVariance, error, r)
                superScoreCount += 1
                superScoreAvg = np.average(superScoreList)

                summary = tf.Summary()
                summary.value.add(tag='1_error', simple_value=error)
                summary.value.add(tag='0_superScore', simple_value=superScoreAvg)
                summary.value.add(tag='2_finalOutVarianceQuota', simple_value=finalOutVarianceQuota)
                summary.value.add(tag='4_errorVariance', simple_value=errorVariance)
                summary.value.add(tag='5_rawQuantError', simple_value=rawQuantError)
                summary.value.add(tag='6_infQuantOutVariance', simple_value=infQuantOutVariance)
                summary.value.add(tag='7_infFinalOutVariance', simple_value=infFinalOutVariance)
                summary.value.add(tag='8_labelFinalOutVariance', simple_value=labelFinalOutVariance)
                summary.value.add(tag='9_trainTimePerSample_us', simple_value=trainTimePerSample_us)
                summary.value.add(tag='9_inferenceTime_ms', simple_value=inferenceTime_ms)
                summary.value.add(tag='9_inferenceTimePerMainLoop_ms', simple_value=inferenceTimePerMainLoop_ms)
                summary.value.add(tag='9_trainTimePerMainLoop_s', simple_value=trainTimePerMainLoop_s)
                self.summary_writer.add_summary(summary, r)

                if superScoreAvg > maxSuperScore:

                    print(f"step: {r} New Superscore: {superScoreAvg:.4f}")
                    maxSuperScore = superScoreAvg

                    #
                    # GENERATE EXAMPLE SOUND and save graph to disk, but not to often...
                    #
                    if soundWriteCounter < (r - self.params['MIN_STEPS_BETWEEN_SAVES']):
                        soundWriteCounter = r

                        print("Saving graph to disk, and saving example sound from inference...")

                        self.saveGraphToDisk()

                        pieceOfInputSound = self.audio.getAPieceOfSound(inputSoundVal, 0, inputSoundVal["sampleCount"])
                        pieceOfLabelSound = self.audio.getAPieceOfSound(labelSoundVal, 0, labelSoundVal["sampleCount"])
                        infOutSound = self.runInferenceOnSoundSampleBySample(pieceOfInputSound)
                        self.audio.writeSoundToDir(infOutSound, defs.WAV_FILE_OUTPUT, self.params["uniqueSessionNumber"] + "-" + str(r))
                        self.plotter.plotSoundSimple(pieceOfInputSound, pieceOfLabelSound, infOutSound, audio4=None, useSame=True, blocking=self.blockNextImgPrintOut)
            #
            # TRAIN
            #

            with self.globalLock:
                inputBatch = np.zeros(shape=(self.params['BATCH_SIZE'],self.params['networkInputLen']))
                labelBatch = np.zeros(shape=(self.params['BATCH_SIZE'],defs.quantSteps))

                for p in range(self.params['BATCH_SIZE']):

                    randomIndex = random.randint(0, len(self.trainingData) - 1)
                    inputBatch[p] = self.trainingData[randomIndex]["input"]
                    if defs.USE_QUANT_OUT:
                        labelBatch[p] = self.trainingData[randomIndex]["quantArrayLabel"]
                    else:
                        labelBatch[p] = self.trainingData[randomIndex]["label"] ## Todo: ok?

            startTime = time.time()

            if self.slowMode:
                time.sleep(5)

            self.train(inputBatch, labelBatch, r)
            trainTime = time.time() - startTime
            trainTimePerSample_us = 1000000* (trainTime / self.params['BATCH_SIZE'])

        self.sessionFC.close()




    #####################################################
    # Function
    #####################################################
    def newConvLayer(self, input, stride, num_input_channels, filter_size, num_filters, use_relu=True, aggregatedStride=1):

        weights = self.new_weights([filter_size, num_input_channels, num_filters])
        biases = self.new_biases(length=num_filters)
        layer = tf.reshape(input, [-1, int(input.shape[1]), num_input_channels]) # batch, inputSize, channels
        layer = tf.nn.conv1d(layer, weights, stride, 'SAME') #Value, filter, stride, padding
        layer += biases

        if use_relu:
            layer = tf.nn.leaky_relu(layer)
        else:
            layer = tf.nn.sigmoid(layer)

        return layer, num_filters, aggregatedStride * stride


    #####################################################
    # Function
    #####################################################
    def new_fc_layer(self,
                     input,          # The previous layer.
                     num_inputs,     # Num. inputs from prev. layer.
                     num_outputs,    # Num. outputs.
                     use_relu=True): # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.leaky_relu(layer)
        else:
            layer = tf.nn.sigmoid(layer)

        return layer

    #####################################################
    # Function
    #####################################################
    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05), dtype=tf.float32)

    #####################################################
    # Function
    #####################################################
    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]), dtype=tf.float32)

    # ############################################################################
    # save variables to disk
    # ############################################################################
    def saveGraphToDisk(self):

        try:
            os.mkdir(defs.GRAPH_PATH)
        except Exception as ex:
            pass

        with self.graphFC.as_default() as g:
            saver = tf.train.Saver()
            saver.save(self.sessionFC, self.fullGraphPath)

    #####################################################
    # learn by history
    #####################################################
    def train(self, X_training, Y_training, iteration):

        assert self.sessionFC is not None and self.graphFC is not None

        feed_dict_batch = {self.xFC: X_training, self.y_true_FC: Y_training}

        with self.graphFC.as_default() as g:

            _, summary = self.sessionFC.run([self.optimizerFC, self.merged_summary_op], feed_dict=feed_dict_batch)

            if iteration%100 == 0:
                self.summary_writer.add_summary(summary, iteration)

    #####################################################
    # Calculates the output from the FC network
    #####################################################
    def getFCOutput(self, dataX):
        assert self.sessionFC is not None and self.graphFC is not None

        feed_dict = {self.xFC: dataX}

        with self.graphFC.as_default() as g:
            return self.sessionFC.run(self.y_modelFC, feed_dict=feed_dict)


# ############################################################################
# Main!
# ############################################################################
if __name__ == '__main__':

    MainTrainer().trainToReproduceSoundOneValueAtTheTime()
