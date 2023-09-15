import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

from utils import review_to_words, convert_and_pad

def model_fn(model_dir):
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])
    
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    dict_size = model_info['vocab_size']
    print("**************************",dict_size, type(dict_size))
    
    dict_file = "word_dict.pkl"
    dict10k_file = "word_dict_10k.pkl"
    dict_name = str([dict_file if dict_size==5000 else dict10k_file][0])
    
    print("model_fn : Using Word Dict size of : {},from file : {}.".format(dict_size,dict_name))    
    word_dict_path = os.path.join(model_dir, dict_name)
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')

    data_X, data_len = convert_and_pad(model.word_dict, review_to_words(input_data))

    data_pack = np.hstack((data_len, data_X))
    data_pack = data_pack.reshape(1, -1)
    
    data = torch.from_numpy(data_pack)
    data = data.to(device)

    model.eval()

    with torch.no_grad():
        output = model.forward(data)
        
    result = np.round(output.numpy()).astype(int)

    return result
