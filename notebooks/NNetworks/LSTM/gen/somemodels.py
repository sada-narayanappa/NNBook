#!/usr/local/bin/python 

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import optimizers

def SimpleModel1(history, nfeatures, nOut, **kwargs) :
    lstm_input = Input(shape=(history, nfeatures), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(nOut, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    
    return model

def SimpleModel2(inps, inshape, units2=None, nsteps=1, opt="adam", loss="mse", bi=False, dropout=None):
    s= inshape
    print(locals())
    print(f"Creating LSTM: inuts= {inps} time-steps: {s[0]}, features: {s[1]} #out: {nsteps}")
    m = keras.models.Sequential()

    if (bi):
        m.add(keras.layers.Bidirectional(
            keras.layers.LSTM(inps, return_sequences= (units2 is not None), input_shape=s) ) )
    else:
        m.add(keras.layers.LSTM(inps, return_sequences= (units2 is not None), input_shape=s) )
    
    if(units2 is not None): #Lets just keep it simple for 2 layers only
        m.add(keras.layers.LSTM(units2, activation='relu'))
    if (dropout is not None):
        m.add( keras.layers.Dropout(dropout) )
    m.add(keras.layers.Dense(nsteps))
    m.compile(optimizer = opt, loss= loss)
    return m

def UberModel(lookBack, nFeatures, lstm_IPDim=256, lstm_OPDim=1, opt=None, loss="mse",  drop=0.3):
    opt        = opt or optimizers.Adam(lr=0.0005)
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

    memory_layer = LSTM (units=lstm_IPDim, 
                             kernel_regularizer = k_rrizer, 
                             recurrent_regularizer = r_rrizer, 
                             return_sequences=False)(concat_layer)
    preds = Dense(units=lstm_OPDim, activation='linear')(memory_layer)

    model1 = Model(input_layer, preds)
    model1.compile(optimizer = opt, loss= loss)             

    return model1
