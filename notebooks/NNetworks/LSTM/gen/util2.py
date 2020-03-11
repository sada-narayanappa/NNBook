#!/usr/local/bin/python 

import matplotlib.pyplot as plt
%matplotlib inline  

%load_ext autoreload
%autoreload 2

import re, sys, os, datetime, getopt, glob, argparse, json, base64, pickle
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (16, 5)
mpl.rcParams['axes.grid'] = False

import tensorflow as tf
tf.random.set_seed(13)
pd.options.display.max_rows = 8

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import optimizers
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, Callback

sys.path.append(".")
sys.path.append("gen")
import utils1;
import ccallbacks

np.random.seed(4)

def getInvertedPreds(conf, yh):
    scalerY = utils1.getConfigObject(conf, "scalerYString")
    sOuputs = utils1.getConfigList(conf, 'scaleOutputs')
    ouputs  = utils1.getConfigList(conf, 'outputs')
    
    yhdf    = pd.DataFrame(yh, columns=ouputs)    # Dataframe of Predictions
    ys      = yhdf[sOuputs].values                # Values to be scaledback
    yi      = scalerY.inverse_transform(ys)       # inverse transform the outputs
    yidf    = yhdf.copy()                         # Copy and set the values
    yidf[sOuputs] = yi
    return yhdf, yidf
    
def getOriginal(conf, unnormdf, index=0):
    inputs  = utils1.getConfigList(conf, 'inputs')
    ouputs  = utils1.getConfigList(conf, 'outputs')
    
    index   = 0
    startIX = index + tsParams['length']
    batchSZ = 1 # batch size
    stride  = tsParams.get('stride', 1)
    i = startIX + batchSZ * stride * index
    return unnormdf[i:], inputs, ouputs

def plotInverted(conf, yh, unnormdf, s =-400, howmany=100):
    e=s+howmany
    
    yhdf, yidf = getInvertedPreds(conf, yh)
    yorg,ips,ops = getOriginal(conf, unnormdf)

    x = pd.to_datetime(yorg[yorg.columns[0]][s:e])
    
    plt.plot(x, yorg[ops].values[s:e], marker='.', label="Original")
    plt.plot(x, yidf.values[s:e], marker='x', label="Predicted")
    plt.title("Plotting Inverted Values:")
    plt.grid()
    plt.legend()

'''
Reconstruct the original diffed columns
'''    
def reconstructOrig(conf, unnormdf, yh):
    yhdf, yidf = getInvertedPreds(conf, yh)
    yorg,ips,ops = getOriginal(conf, unnormdf)
    
    for o in ouputs:
        if(o.endswith("___diff1")):
            oc = o[:-8]
            print(f"Getting Original column for: '{oc}' ")
            if ( oc not in yorg.columns):
                print('Cannot compute the orginal column values from diffs for {oc}')
                continue;
                
            ## WOW <== this is heavy - undo the diffing in the opposite way
            yidf[oc] = yorg[oc].values + yidf[o].shift(-1)

    return yidf # y inverted dataframe with adjusted cols for diffs

yh = ccallbacks.predict(modelFile, valg2, model)
yidf = reconstructOrig(conf, unnormdf, yh)
