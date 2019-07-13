

class Definitions:

    USE_QUANT_OUT=False
    WAV_FILE_PATH = 'soundStorage'
    WAV_FILE_OUTPUT = 'output'
    METADATA_OUTPUT = 'metadata'
    GRAPH_PATH = 'graph'
    TENSORBOARD_PATH = 'tensorboard'
    FFTAudioWindowLength = 32


    if USE_QUANT_OUT:
        quantSteps = 512
    else:
        quantSteps = 1 # Should always be 1!

    sampleToPredict = -1  # -1 = lastOne
