import warnings, logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pickle


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def cifar10_data(num_classes):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test
