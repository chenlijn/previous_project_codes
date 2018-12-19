import keras
import os
from keras.models import load_model
from keras import backend as K
import tensorflow as tf


def save_tf_graph_and_weights(model_path, graph_dir, ckpt_path):
    model = load_model(model_path)
    K.set_learning_phase(0)
    saver = tf.train.Saver()
    for layer in model.layers[:3]:
       print(layer)
       print(layer.name) 
       print(layer.input)
       print(layer.input.name)
       print(layer.output.name)
    #with K.get_session() as sess:
    #    tf.train.write_graph(sess.graph_def, graph_dir, 'inception-v3.pbtxt')
    #    saver.save(sess, ckpt_path)


def freeze_the_graph(graph_path, ckpt_path):
    from tensorflow.python.tools import freeze_graph
    model_name = os.path.dirname(graph_path) + '/' + graph_path.split('/')[-1].split('.')[0] + "_frozen"
    input_graph_path = graph_path
    checkpoint_path = ckpt_path
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "dense_2/Softmax" 
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = model_name + '.pb'
    clear_devices = True
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary,
                              checkpoint_path, output_node_names, restore_op_name,
                              filename_tensor_name, output_frozen_graph_name, clear_devices, "")

if __name__=="__main__":

   model_path = "saved_models/incept-v3_whole_trained.h5"
   graph_dir = "frozen_model"
   ckpt_path = "tf_ckpt/incept_v3"

   save_tf_graph_and_weights(model_path, graph_dir, ckpt_path)
   #freeze_the_graph(graph_dir+"/inception-v3.pbtxt", ckpt_path)

