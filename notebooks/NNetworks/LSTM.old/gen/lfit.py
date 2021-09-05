#!/usr/local/bin/python 

import matplotlib.pyplot as plt

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

def getConf(cfile = "myconfig"):
    conf=utils1.getconfig(cfile)
    trnFile       = conf['normalizedFile']
    orgFile       = conf['unnormalizedFile']

    ddir = ""
    if not os.path.exists(trnFile):
        ddir = os.path.abspath(cfile)
        ddir = os.path.dirname(ddir) 
        trnFile       = f'{ddir}/{trnFile}'
        orgFile       = f'{ddir}/{orgFile}'

    normeddf     = pd.read_csv(trnFile)
    unnormdf     = pd.read_csv(orgFile)
    inputs       = utils1.getConfigList(conf, 'inputs')
    ouputs       = utils1.getConfigList(conf, 'outputs')
    train_pct    = conf.get('train_pct', 0.9)
    train_count  = conf.get('train_count', int(len(normeddf) * train_pct) )

    print(f'''
    TrnFile: {trnFile},
    I/P    : {inputs[0:4]} ...
    O/P    : {ouputs[0:4]} ...
    Shape  : {normeddf.shape}
    trnCnt : {train_count}"
    ''')
    
    return conf, unnormdf, normeddf, inputs, ouputs

def getGenerators(conf, normeddf, inputs, ouputs):
    modelFile    = conf['modelFile'] or "models/simpleModel.h5"
    tsParams     = conf['tsParams']
    lookahead    = conf['lookahead']
    history      = tsParams['length']
    train_pct    = conf.get('train_pct', 0.9)
    train_count  = conf.get('train_count', int(len(normeddf) * train_pct) )

    X, y = normeddf[inputs].values, normeddf[ouputs].values
    X=X[:(-lookahead+1) or None]
    y=y[lookahead-1:]

    Xtrn,ytrn = X[:train_count], y[:train_count], 
    Xtst,ytst = X[train_count:], y[train_count:], 

    tsParams1 = tsParams.copy()
    tsParams2 = tsParams.copy()
    tsParams2['batch_size'] =1

    trng1 = TimeseriesGenerator(Xtrn, ytrn, **tsParams1 )
    valg1 = TimeseriesGenerator(Xtst, ytst, **tsParams1 )
    valg2 = TimeseriesGenerator(X, y, **tsParams2 )

    #history, tsParams1, len(trng1), len(valg1), len(valg2), #trng1[0]
    #print(Xtrn.shape, "\n", Xtrn, "\n", ytrn.shape, "\n", ytrn, Xtst.shape)

    return modelFile, history, lookahead, trng1, valg1, valg2, X, y

def getModel(conf):
    modelFile = conf['modelFile']
    modelName = conf['modelName']
    loadModel = conf['loadModel']
    tsParams  = conf['tsParams']
    
    importName= modelName.split(".")[0]
    
    print(f'''
    import: {importName}
    using : {modelName}
    saved : {modelFile}
    reload: {loadModel}
    ''')
    
    exec(f"import {importName}")

    mcp= ccallbacks.ModelCheckAndLoad(modelFile, 'val_loss', best=np.inf, stop_at=False, verbose=1)
    if ( loadModel and os.path.exists(modelFile)):
        m1 = mcp.load_ext()
    if ( m1 is None):
        m1=eval(modelName)
    m1.summary()
        
    return m1, mcp


def fit(model, trng1, valg1, mcpoint,validation_steps=50, vv =0, ep = 1, spe =200 ):
    model.fit(trng1, verbose=vv, epochs=ep, validation_data=valg1,steps_per_epoch=spe, shuffle=True, 
                        validation_steps=validation_steps, callbacks=[mcpoint])

