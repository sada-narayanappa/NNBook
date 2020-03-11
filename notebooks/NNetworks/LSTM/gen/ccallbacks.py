#!/usr/local/bin/python 

import matplotlib.pyplot as plt
import re, sys, os, datetime, glob, json, base64, pickle, sklearn
import pandas as pd
import numpy as np

import keras
from keras.models import Model
from keras.models import load_model
from keras.callbacks import Callback
import IPython
from IPython.display import display

sys.path.append(".")
sys.path.append("gen")
import utils1;
import sklearn.metrics

class ModelCheckAndLoad(Callback):
    def __init__(self, filepath, monitor='val_loss', best=np.inf, 
                 stop_at=False, verbose=0, drawLoss=False):
        super(Callback, self).__init__()
        self.monitor  = monitor
        self.filepath = filepath
        self.verbose  = verbose
        self.best     = best or np.inf
        self.stop_at  = stop_at;
        self.history  = {}
        self.epochs   = []
        self.drawLoss = drawLoss
        self.epochNum = 0
        
    def save_ext(self):
        ef = self.filepath+"_ext"
        with open(ef, "wb") as f:
            myParams = {
                'best'     : self.best,
                'epochNum' : self.epochNum,
                'history'  : self.history
            }
            pickle.dump(myParams, f, protocol=pickle.HIGHEST_PROTOCOL)
                
    def load_ext(self):
        ret = None;
        if ( os.path.exists(self.filepath)):
            ret = load_model(self.filepath)
        
        ef = self.filepath+"_ext"
        if ( not os.path.exists(ef) or os.path.getsize(ef) <= 0):
            return ret
        
        with open(ef, "rb") as f:
            myParams      = pickle.load(f)
            self.best     = myParams.get('best'    , np.inf)
            self.epochNum = myParams.get('epochNum', 0);
            self.history  = myParams.get('history', {});
            
        print(f"Best Loaded {self.best}")
        return ret;

    def drawLosses(self):
        history, best = self.history, self.best
        #IPython.display.clear_output(wait=True)
        plt.clf()

        fig, ax1 = plt.subplots()
        i, colors, marks = 0, "rgbcmykw", "v.xo+"

        color = colors[i]
        ax1.set_xlabel('epochs')
        k, v = "loss", history['loss']
        ax1.set_ylabel(k, color=color)
        l1= ax1.plot(v, color=color, marker=marks[i], label=f"{k}")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        i +=1
        k, v = "val_loss", history['val_loss']
        color = colors[i]
        ax2.set_ylabel(k, color=color)
        l2 = ax2.plot(v, color=color, marker=marks[i], label=f"{k}")

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        l3 = plt.plot(0, best, marker="o",  c="b", label=f"BEST: {best}")
        ax1.grid()

        lns  = l1 + l2 + l3;
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=0)
        plt.show()
        
    def on_epoch_end(self, epoch, logs={}):
        self.epochs.append(epoch)
        self.epochNum += 1;
        
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        self.current = logs.get(self.monitor)
        if self.current is None:
            warnings.warn(f'Can save best model only with {self.monitor} available')
            return;
                    
        if (self.best > self.current):
            ou= f'{self.monitor}: {self.best} > {self.current}\n'
            print(f"Epoch: {epoch+1} Saving: {ou}");
            
            self.best = self.current
            self.model.save(self.filepath, overwrite=True)
            self.save_ext();
            self.model.stop_training = self.stop_at
        elif self.verbose > 0:
            ou= f'{self.monitor}: {self.best} <= {self.current}'
            print(f"{epoch+1} din't improve : {ou}\r", end="")
            
        if (self.drawLoss):
            drawLosses(self.history, self.best)

def predict(modelFile, valg, model=None):
    m1 = model or load_model(modelFile)
    #xxt = np.array([valg[i][0][0] for i in range(len(valg))])
    #yyt = np.array([valg[i][1][0] for i in range(len(valg))])
    yh=m1.predict(valg)
    return yh
            
def plot1(modelFile, valg, model=None, idx=0, n=-150, howmany=50):
    yh = predict(modelFile, valg, model)
    yy = np.array([valg[i][1][idx] for i in range(len(valg))])
                 
    plt.gcf().set_size_inches(22, 10, forward=True)
    plt.plot( yy[n:n+howmany], marker='o', label="original-")
    plt.plot( yh[n:n+howmany], marker='x', label="predicted")
    mse = sklearn.metrics.mean_squared_error(yy, yh)
    
    plt.title(f"{modelFile} : {model}: MSE: {mse} <==")
    plt.grid()
    plt.legend()
    plt.show()
    
    return yy, yh, mse
