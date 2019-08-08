import os
from definitions import Definitions as defs
from tensorflow.python.tools import freeze_graph

# Freeze the graph
checkpoint_state_name = "checkpoint"

input_saver_def_path = ""
input_binary = False
input_checkpoint_path = defs.fullGraphPath
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
clear_devices = True

freeze_graph.freeze_graph(defs.input_graph_name, input_saver_def_path,
                          input_binary, input_checkpoint_path,
                          defs.output_node_names, restore_op_name,
                          filename_tensor_name, defs.output_graph_name,
                          clear_devices,"")



