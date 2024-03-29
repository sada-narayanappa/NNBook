#!/usr/local/bin/python 

import re, sys, os, datetime, getopt, glob, argparse, datetime, json, base64, pickle, glob
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
sys.path.append(".")
sys.path.append("gen")
import dataconfig;
import keras
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (16, 5)
mpl.rcParams['axes.grid'] = False
pd.options.display.max_rows = 5
%matplotlib inline  

use_keras=1
if ( use_keras):
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, TimeDistributed
    from keras.layers import Conv1D, GlobalMaxPool1D,Flatten, Bidirectional, RepeatVector, MaxPooling1D
    from keras.preprocessing.sequence import TimeseriesGenerator
    from keras import regularizers
    from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
    from keras import optimizers
    from keras.models import load_model
else:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, TimeDistributed
    from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D,Flatten, Bidirectional, RepeatVector, MaxPooling1D
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import load_model

#-----------------------------------------------------------------------------------
def getJSONconfig(cf = "config*"):
    with open(cf, "r") as f:
        cf = f.read()

    r1=re.findall("\[START](.*)\[END]", cf, flags=re.MULTILINE|re.DOTALL)
    if ( len(r1) <= 0):
        print(f"Ignoring: Configuration not found in {cf}! no worries")
        return None
    r1 = r1[0].replace("'", '"')    
    rj = eval(r1)
    return rj
#-----------------------------------------------------------------------------------
def process(config, input_files, outputFile=None):
    conf = getconfig(config)
    
    model1=None      

    #sensN = len(self.train_transformed[0].columns)  # number of sensors (eliminating the two time ones)
    #outN = len(self.num_id_list) # number of output sensors; the non-categorical ones        

    lookBack   = tsParam['length']
    nFeatures  = X.shape[1]  # Number of features 
    lstm_OPDim = y.shape[1]  # This is usually all sensors except categorical that to train LSTM on
    lstm_IPDim = 256
    drop       = 0.3
    optimizer  = optimizers.Adam(lr=0.0005)
    loss       = 'mse'
    k_rrizer   = None
    r_rrizer   = None

    input_layer  = Input(shape=(lookBack, nFeatures), dtype='float32', name='input')
    memory_layer = LSTM( lstm_IPDim, return_sequences=True, name="memory1")(input_layer)
    memory_layer = LSTM (int(lstm_IPDim/2), return_sequences=False, name="memory2")(memory_layer)
    repeated     = RepeatVector(lookBack)(memory_layer)
    memory_layer = LSTM (int(lstm_IPDim/2), return_sequences=True, name="first1out")(repeated)
    memory_layer = LSTM (lstm_IPDim,  return_sequences=True, name="first2out")(memory_layer)
    decoded_inputs = TimeDistributed(Dense(units=lstm_OPDim, activation='linear'))( memory_layer)

    #  Try spatial dropout?
    dropout_input = Dropout(drop)(input_layer)
    concat_layer  = concatenate([dropout_input, decoded_inputs])

    #memory_layer = LSTM (units=self.lstm_dim, return_sequences=False)(concat_layer)
    memory_layer = LSTM (units=lstm_IPDim, 
                             kernel_regularizer = k_rrizer, 
                             recurrent_regularizer = r_rrizer, 
                             return_sequences=False)(concat_layer)
    preds = Dense(units=lstm_OPDim, activation='linear')(memory_layer)

    model1 = Model(input_layer, preds)
    model1.compile(optimizer = optimizer, loss= loss)             

    print(model1.summary())
    return conf, df, adf;

#-----------------------------------------------------------------------------------
sysargs=None
def addargs():
    sysargs = None
    p = argparse.ArgumentParser(f"{os.path.basename(sys.argv[0])}:")
    p.add_argument('-c', '--config', type=str, default="config.txt", help="Config Files")
    p.add_argument('-o', '--output', type=str, default="out.csv", help="output file")
    p.add_argument('args', nargs=argparse.REMAINDER)
    p.add_argument('input_files',action="store", type=str, nargs='+', help="input file(s)")

    #p.print_help() # always print -help
    try:
        sysargs=p.parse_args(sys.argv[1:])
    except argparse.ArgumentError as exc:
        print(exc.message )
        
    return sysargs
#-----------------------------------------------------------------------------------
#*** NOTE: DO NOT EDIT THIS FILE - THIS iS CREATED FROM: inv_utils.ipynb
def inJupyter():
    try:    get_ipython; return True
    except: return False

#-----------------------------------------------------------------------------------
if __name__ == '__main__':
    if (not inJupyter()):
        t1 = datetime.datetime.now()
        sysargs = addargs()
        ret = process(sysargs.config, sysargs.input_files, sysargs.output)
        t2 = datetime.datetime.now()
        print(f"#All Done in {str(t2-t1)} ***")
    else:
        pass
