import common.audioPlotter as audioPlotter
import common.audioHandler as audioHandler
from common.definitions import Definitions as defs
import common.network as network
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
                per_process_gpu_memory_fraction = 0.20,
                USE_RELU = True,  # False => sigmoid
                BATCH_SIZE = 400,  # Batch size for training.
                BATCH_INF_SIZE = 75,  # How many samples for stats purpose
                BATCH_SIZE_INFERENCE_FULL_SOUND = 1024,  # Batch size (#samples!!) When running inference to generate full audio file
                STATS_EVERY = 250,  # How often (skipping steps) to run inference to gather stats.
                validationPercent = 0.40,  # e.g. 0.1 means 10% of the length of total sound will be validation
                maxValidationSampleCount = 1500000,
                MIN_STEPS_BETWEEN_SAVES = 6000,
                learning_rate = 0.0002,
                learning_rate_decay = 6000000 , # Higher gives slower decay
                networkInputLen = defs.networkInputLen,
                networkOutputLen = defs.networkOutputLen,
                encoderBullsEyeSize = 55,
                graphName = 'latest',
                maxTrainingSamplesInMem=250000,
                effectiveInferenceOutputLen = 128,
                inferenceOverlap = 8,
                lowPassFilterSteps = 0,
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
            'maxValidationSampleCount' : maxValidationSampleCount,
            'MIN_STEPS_BETWEEN_SAVES' : MIN_STEPS_BETWEEN_SAVES,
            'learning_rate' : learning_rate,
            'encoderBullsEyeSize' : encoderBullsEyeSize,
            'networkInputLen' : networkInputLen,
            'networkOutputLen' : networkOutputLen,
            'graphName': graphName,
            'uniqueSessionNumber' : uniqueSessionNumber,
            'maxTrainingSamplesInMem': maxTrainingSamplesInMem,
            'inferenceOverlap' : inferenceOverlap,
            'effectiveInferenceOutputLen': effectiveInferenceOutputLen,
            'lowPassFilterSteps': lowPassFilterSteps
        }
        print(f"Unique session number: {self.params['uniqueSessionNumber']}")
        self.totalVariablesCount = 0
        self.tensorboardFullPath = os.path.join(defs.TENSORBOARD_PATH, self.params['uniqueSessionNumber'])
        self.graphFC, self.sessionFC, self.xFC, self.y_modelFC = network.defineFCModel(
            self.params['networkInputLen'], self.params['networkOutputLen'],
            per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction)
        self.y_true_FC, self.optimizerFCALL, self.merged_summary_op, self.summary_writer = self.trainingSetup(
            self.y_modelFC, self.graphFC,  self.sessionFC, defs.fullGraphPath,
            self.tensorboardFullPath, self.params['networkOutputLen'],
            self.params['learning_rate'], self.params['learning_rate_decay'])
        self.audio = audioHandler.AudioHandler()
        self.plotter = audioPlotter.AudioPlotter()
        self.trainingData = []
        self.validationData = []
        self.globalLock = Lock()
        self.blockNextImgPrintOut = False
        self.slowMode = False
        self.printCounter = 0

        print(f"output_node_name for frozen graph: {self.y_modelFC.name.split(':')[0]}")

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

    #####################################################
    # keyboard input...
    #####################################################
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
    # Set up optimizers, tensorboard, restore graphs...
    #####################################################
    def trainingSetup(self, y_modelFC, graphFC, sessionFC, fullGraphPath, tensorboardFullPath, networkOutputLen, learning_rate, learning_rate_decay):

        with graphFC.as_default() as g:

            # cost functions and optimizers..
            retValy_true_FC = tf.placeholder(tf.float32, shape=[None, networkOutputLen], name='y_trueFC')
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate, global_step, learning_rate_decay, 0.99,
                                                       staircase=True)
            cost = tf.reduce_mean(tf.square(y_modelFC - retValy_true_FC))

            # Probably meaningsless gradient clipping by norm. But it cant really hurt the sound quality..
            #opt_func = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            #tvars = tf.trainable_variables()
            #grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 1.0)
            #retValoptimizerFC = opt_func.apply_gradients(zip(grads, tvars))

            tvars = tf.trainable_variables()
            PHASEALL_vars = tvars

            retValoptimizerFCALL = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step, var_list=PHASEALL_vars)

            # Tensorboard
            tensor_summaries_list = []
            for tvar in tvars:
                tensor_summaries_list.append(tf.summary.histogram(tvar.name, tvar))
            tf.summary.merge(tensor_summaries_list)
            tf.summary.scalar("1_loss", cost)
            tf.summary.scalar("3_learning_rate", learning_rate)
            retValmerged_summary_op = tf.summary.merge_all()
            retValsummary_writer = tf.summary.FileWriter(tensorboardFullPath, sessionFC.graph)

            network.restoreGraphFromDisk(sessionFC, self.graphFC, fullGraphPath)
            self.totalVariablesCount = self.countTotalWeights()
            return retValy_true_FC, retValoptimizerFCALL, retValmerged_summary_op, retValsummary_writer


    #####################################################
    # Count the number of variables in the network
    #####################################################
    def countTotalWeights(self):
        t_vars = tf.trainable_variables()
        totResult = 0

        for next in t_vars:
            result = 1
            for y in range(int(len(next.shape))):
                result *= int(next.shape[y])

            print(f"Size {next.name}: {result}")
            totResult += result

        print(f"Total variables in network: {totResult}")
        return totResult


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
                label = np.array(nextLabelDataSlize["scaledData"][-self.params['networkOutputLen']:]).reshape(self.params['networkOutputLen'])
                quantArrayLabel = label

                reshapedInput = input.reshape(-1, self.params['networkInputLen'])
                reshapedLabel = label.reshape(-1, self.params['networkOutputLen'])
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
                time.sleep(30) # Take a little break until we continue to replace the training data with new samples.

    #####################################################
    # Calculates a score that means the higher the better
    # and should "always" be comparable... You need variance
    # similar to label variance in order for a sound to be
    # good, in combinatino with a low error value.
    #####################################################
    def calcSuperScore(self, labelVar, infVar, error, fftDiffScore, step):

        if step < 5:
            return 0.0, 0.0

        if infVar < 0.00000001:
            infVar  = 0.00000001
        if labelVar < 0.00000001:
            labelVar = 0.00000001

        retVal = 0.0

        try:
            if (labelVar >= infVar): # varDiff always >= 1, where closest to 1 is the best!
                varDiff = math.fabs(labelVar / infVar)
            else:
                varDiff = math.fabs(infVar / labelVar)

            errSqTimesDiff = error * varDiff * fftDiffScore
            if (errSqTimesDiff) < 0.00000001:
                errSqTimesDiff = 0.00000001
            retVal = 100 / errSqTimesDiff
        except Exception as ex:
            pass

        return retVal, (retVal * 2000000/self.totalVariablesCount)


    #####################################################
    # Train a network
    #####################################################
    def trainMain(self, slicesPerLoop = 250):

        trainTimePerSample_us = 0
        maxSuperScore = -999999.9
        soundWriteCounter = -10000000
        superScoreList = np.zeros(shape=10)
        superScoreListV3 = np.zeros(shape=10)
        superScoreCount = 0
        inputSoundRaw = None
        labelSoundRaw = None
        fftDiffScore = 20000

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
        inputSoundVal = self.audio.getAPieceOfSound(inputSoundRaw, validationStartPoint, self.params['maxValidationSampleCount'])#validationSampleCount'])
        labelSoundVal = self.audio.getAPieceOfSound(labelSoundRaw, validationStartPoint, self.params['maxValidationSampleCount'])#validationSampleCount'])

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
        #self.plotter.plotSoundSimple(labelSoundVal, None, None, None, False, True)

        for r in range(11111111):

            if r < 0:
                sameOutput = True
            else:
                sameOutput = False

            if r % math.floor(self.params['STATS_EVERY']) is 0:

                #
                # GREATE METRICS TO TENSORBOARD
                #
                error = 0
                generatorPrecisionError = 0
                startTime = time.time()


                with self.globalLock:
                    inputBatch = np.zeros(shape=(self.params['BATCH_INF_SIZE'], self.params['networkInputLen']))
                    labelBatch = np.zeros(shape=(self.params['BATCH_INF_SIZE'], self.params['networkOutputLen']))
                    finalOutInfVariance = np.zeros(shape=(self.params['BATCH_INF_SIZE'] * self.params['networkOutputLen']))
                    finalOutLabelVariance = np.zeros(shape=(self.params['BATCH_INF_SIZE'] * self.params['networkOutputLen']))

                    for p in range(self.params['BATCH_INF_SIZE']):
                        randomIndex = random.randint(0, len(self.validationData) - 1)
                        inputBatch[p] = self.validationData[randomIndex]["input"]
                        labelBatch[p] = self.validationData[randomIndex]["label"]

                    outputArray = network.getFCOutput(inputBatch, self.sessionFC, self.graphFC, self.xFC, self.y_modelFC)

                    ccc = 0

                    for index, nextOutput in enumerate(outputArray):
                        tmpGeneratorPrecisionVariance = np.zeros(shape=(self.params['networkOutputLen']))
                        for y in range(self.params['networkOutputLen']):
                            infOutput = nextOutput[y]
                            tmpGeneratorPrecisionVariance[y] += infOutput
                            labelOutput = labelBatch[index][y]
                            finalOutInfVariance[ccc] = infOutput
                            finalOutLabelVariance[ccc] = labelOutput
                            lastErr = 0.95 * ((math.fabs(infOutput - labelOutput)) / self.audio.K)  # puhh. 0.95 because I wantto compare to old scores....
                            error += lastErr
                            ccc += 1

                        generatorPrecisionError += np.var(tmpGeneratorPrecisionVariance)

                # Run a shorter inference piece in order to get a error of the FFT over the label vs inference
                pieceOfInputSound = self.audio.getAPieceOfSound(inputSoundVal, 0, defs.FFT_DiffLength)
                pieceOfLabelSound = self.audio.getAPieceOfSound(labelSoundVal, 0, defs.FFT_DiffLength)
                infOutSound, infTime = network.runInferenceOnSoundSampleBySample(pieceOfInputSound, self.audio, self.params['networkInputLen'],
                                                                        self.params['networkOutputLen'], self.params['inferenceOverlap'],
                                                                        self.params['effectiveInferenceOutputLen'],
                                                                        self.params['BATCH_SIZE_INFERENCE_FULL_SOUND'],
                                                                        self.sessionFC, self.graphFC, self.xFC, self.y_modelFC)
                fftDiffScore = self.audio.soundDiffFFT(infOutSound, pieceOfLabelSound)

                infTime = time.time() - startTime
                inferenceTime_ms = 1000 * infTime
                inferenceTimePerMainLoop_ms = inferenceTime_ms / self.params['STATS_EVERY']
                trainTimePerMainLoop_s = trainTimePerSample_us * self.params['BATCH_SIZE'] / 1000000
                error /= self.params['BATCH_INF_SIZE'] * self.params['networkOutputLen']
                infFinalOutVariance = np.var(finalOutInfVariance)
                labelFinalOutVariance = np.var(finalOutLabelVariance)
                finalOutVarianceQuota = infFinalOutVariance / labelFinalOutVariance
                generatorPrecisionError /= self.params['BATCH_INF_SIZE']

                superV1, superV3 = self.calcSuperScore(labelFinalOutVariance, infFinalOutVariance, error, fftDiffScore, r)
                superScoreList[superScoreCount % len(superScoreList)] = superV1
                superScoreListV3[superScoreCount % len(superScoreListV3)] = superV3
                superScoreCount += 1
                superScoreAvg = np.average(superScoreList)
                superScoreAvgV3 = np.average(superScoreListV3)

                summary = tf.Summary()

                if r > 1000:
                    summary.value.add(tag='7_infFinalOutVariance', simple_value=infFinalOutVariance)
                    summary.value.add(tag='2_finalOutVarianceQuota', simple_value=finalOutVarianceQuota)
                    summary.value.add(tag='1_FFT_DiffScore', simple_value=fftDiffScore)
                    summary.value.add(tag='1_error', simple_value=error)

                summary.value.add(tag='0_superScoreV2-soundQuality', simple_value=superScoreAvg)
                summary.value.add(tag='0_superScoreV3-networkEfficiency', simple_value=superScoreAvgV3)
                summary.value.add(tag='8_labelFinalOutVariance', simple_value=labelFinalOutVariance)
                summary.value.add(tag='9_trainTimePerSample_us', simple_value=trainTimePerSample_us)
                summary.value.add(tag='9_inferenceTime_ms', simple_value=inferenceTime_ms)
                summary.value.add(tag='9_inferenceTimePerMainLoop_ms', simple_value=inferenceTimePerMainLoop_ms)
                summary.value.add(tag='9_trainTimePerMainLoop_s', simple_value=trainTimePerMainLoop_s)
                self.summary_writer.add_summary(summary, r)

                if superScoreAvg > maxSuperScore:

                    print(f"step: {r} New Superscore: {superScoreAvg:.6f}")
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
                        infOutSound, infTime = network.runInferenceOnSoundSampleBySample(pieceOfInputSound, self.audio, self.params['networkInputLen'],
                                                                        self.params['networkOutputLen'], self.params['inferenceOverlap'],
                                                                        self.params['effectiveInferenceOutputLen'],
                                                                        self.params['BATCH_SIZE_INFERENCE_FULL_SOUND'],
                                                                        self.sessionFC, self.graphFC, self.xFC, self.y_modelFC)
                        self.audio.writeSoundToDir(infOutSound, defs.WAV_FILE_OUTPUT, self.params["uniqueSessionNumber"] + "-" + str(r))
                        self.plotter.plotSoundSimple(pieceOfInputSound, pieceOfLabelSound, infOutSound, audio4=None, useSame=True, blocking=self.blockNextImgPrintOut)
            #
            # TRAIN
            #
            with self.globalLock:
                inputBatch = np.zeros(shape=(self.params['BATCH_SIZE'],self.params['networkInputLen']))
                labelBatch = np.zeros(shape=(self.params['BATCH_SIZE'],self.params['networkOutputLen']))

                for p in range(self.params['BATCH_SIZE']):

                    randomIndex = random.randint(0, len(self.trainingData) - 1)
                    inputBatch[p] = self.trainingData[randomIndex]["input"]
                    labelBatch[p] = self.trainingData[randomIndex]["label"]

            if sameOutput:
                # Re-write the training data so that all outputs are the same..
                for nextSampleData in labelBatch:
                    firstSample = nextSampleData[0]
                    for ix in range(len(nextSampleData)):
                        nextSampleData[ix] = firstSample


            startTime = time.time()

            if self.slowMode:
                time.sleep(5)

            optimizer = self.optimizerFCALL

            self.train(inputBatch, labelBatch, r, optimizer)
            trainTime = time.time() - startTime
            trainTimePerSample_us = 1000000* (trainTime / self.params['BATCH_SIZE'])

        self.sessionFC.close()

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
            saver.save(self.sessionFC, defs.fullGraphPath)
            tf.train.write_graph(self.sessionFC.graph_def, defs.GRAPH_PATH, "graph.pb")

    #####################################################
    # learn by history
    #####################################################
    def train(self, X_training, Y_training, iteration, optimizer):

        assert self.sessionFC is not None and self.graphFC is not None

        feed_dict_batch = {self.xFC: X_training, self.y_true_FC: Y_training}

        with self.graphFC.as_default() as g:

            _, summary = self.sessionFC.run([optimizer, self.merged_summary_op], feed_dict=feed_dict_batch)

            if iteration%100 == 0:
                self.summary_writer.add_summary(summary, iteration)




# ############################################################################
# Main!
# ############################################################################
if __name__ == '__main__':

    MainTrainer().trainMain()
