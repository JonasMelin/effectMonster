import os

class Definitions:

    networkInputLen = 1024
    networkOutputLen = 128
    WAV_FILE_PATH = 'soundInput'
    WAV_FILE_OUTPUT = 'output'
    METADATA_OUTPUT = 'metadata'
    GRAPH_PATH = 'graph'
    TENSORBOARD_PATH = 'tensorboard'
    input_graph_name = os.path.join(GRAPH_PATH, "graph.pb")
    output_graph_name = os.path.join(GRAPH_PATH, "frozen_graph.pb")
    fullGraphPath = os.path.join(GRAPH_PATH, 'latest')
    FFTAudioWindowLength = 32
    FFT_DiffLength = 20000
    output_node_names = "fully_connected_1/Tanh"


