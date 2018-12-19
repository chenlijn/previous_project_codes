
import sys
import os
import numpy as np

sys.path.append("/home/gaia/lijian/utility/") 

from kan_utils import kan_data_proc


dst_dir = "./train_val/"
#input_file = "optic_axis_not_involved_labeled.txt"
#input_file = "light_opaque_labeled.txt"
input_file = "complete_occlusion_labeled.txt"

kan_data_proc.split_train_val_test(input_file, dst_dir)

