import re
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import sklearn
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import r2_score
import time

from keras.layers import Conv1D, GlobalMaxPool1D, Dense, Flatten, LSTM, CuDNNLSTM, Bidirectional, RepeatVector, MaxPooling1D, Dropout, Input, TimeDistributed, concatenate
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint




class anom_lstm():
    """
    
    """
    def __init__(self,
                 train,
                 trainfile = 'Unknown',
                 windowL = 30,
                 cat_num = 5,
                 cat_sensors = [],
                 omit_sensors = [],
                 Verbose=True,
                 oldPred=True):
        """
        Inputs:
        train = list of training dataframes.
        trainfile = path to a training dataset
        cat_list = list of categorical variables (no option now to filter by number of unique value.
        """
#         nonconst = train.nunique()[train.nunique() > 1].index
        self.cat_sensors = cat_sensors
    
        # The list of training dataframes
        self.train_raw = []
        if not type(train) is list:
            train = [train]
        for df in train:
            currdf = df.copy()
            currdf.drop(omit_sensors, axis=1, inplace=True)
            self.train_raw.append(currdf)
        currdf = None
            
        # Eliminate constant sensors and get all training sets to have the same sensors.           
        full_traindf = pd.concat(self.train_raw, axis=0, ignore_index=True)
        full_traindf.fillna(method='ffill', inplace=True)
        full_traindf.fillna(method='bfill', inplace=True)
        nonconstant = full_traindf.nunique()[full_traindf.nunique() > 1].index
        self.train_raw = [df[nonconstant] for df in self.train_raw]
        self.full_traindf = full_traindf[nonconstant] # The dataframe of all training data.  To be used in preprocessing.
        

        
#         self.train_orig = train[nonconst] #raw training and validation data
#         self.train_orig.drop(omit_sensors, axis=1, inplace=True)
        self.trainfile = trainfile
        self.cat_num = cat_num # num unique values necesasry not to be categorical
        self.windowL = windowL # window length to use for one-ahead predictions
        self.quality_sensors = [x for x in self.train_raw[0].columns if x !='time'] # All sensors are assumed high quality at first.
        self.train_loss = []
        self.val_loss = []
        self.train_start = self.train_raw[0].time.iloc[0]
#         self.year_start = year_start
        self.Verbose = Verbose
        self.oldPred = oldPred # Tells if a new prediction is needed (eg. after new data or more training)

    def process_train(self, bats = 256):
        self.batch_size = bats
#         self.train_ratio = train_ratio
#         self.train_val_ratio = train_val_ratio
#         self.train_n = int(len(self.train_orig)*train_ratio)
#         self.val_n = len(self.train_orig) - self.train_n
#         n = len(self.train_orig)
#         self.train_raw = self.train_orig.iloc[:int(n*train_ratio)]
#         self.validate_raw = self.train_orig.iloc[int(n*train_ratio):]
        
        self.train_pred = False # Whether or not predictions have been made on the training set
        # Eliminate constant sensors
        self.ids = list(self.train_raw[0].columns)
        
        # Find any sensor highly correlated with time.
        self.timeSensors = ['time'] #This list in needed below even if we do not check for correlation with time.
#         timeser = pd.Series(self.train_raw[['time']].values.reshape(-1))
#         for sensor in self.ids:
#             sensorSeries = pd.Series(self.train_raw[sensor].values.reshape(-1))
#             if np.abs(timeser.corr(sensorSeries)) >= 0.9:
#                 self.timeSensors.append(sensor)
        
#         # Difference the sensors highly correlated with time
#         timedf = pd.DataFrame()
#         timedf['time'] = self.train_raw.time
#         for sensor in self.timeSensors:
#             if sensor == 'time':
#                 continue
#             parts = re.split(r'\_\_',sensor) #THIS IS FOR JCSAT NAMING CONVENTIONS
# #             parts = re.split(r':',sensor) #THIS IS FOR AM10 NAMING CONVENTIONS
#             diffname = parts[0]+'_DIFF__'+parts[1]
#             timedf[diffname]=self.train_raw[sensor].diff()
#         timedf.fillna(method='bfill', inplace=True)

#         # Checking for difference still correlated with time
#         ids = list(timedf.columns)
#         self.difftimeSensors = []
#         timeser = pd.Series(timedf[['time']].values.reshape(-1))
#         for sensor in ids:
#             sensorSeries = pd.Series(timedf[sensor].values.reshape(-1))
#             if np.abs(timeser.corr(sensorSeries)) >= 0.9:
#                 self.difftimeSensors.append(sensor)
#         # Drop diff'ed sensors correlated with time (and drop time)
#         timedf.drop(self.difftimeSensors, axis=1, inplace=True)
        
    
    
        # Make a scaler that one-hot encodes categorical sensors and scales the others. This should
        # probably be fit on the Orig dataframes with 'time' dropped.
#         unique_vals = self.train_raw.nunique()
        self.cat_sensors = [x for x in list(set(self.ids)-set(self.timeSensors)) if x in self.cat_sensors]
        self.num_sensors = [x for x in list(set(self.ids)-set(self.timeSensors)) if not x in self.cat_sensors]
        #self.cat_sensors = [x for x in list(set(self.ids)-set(self.timeSensors)) if unique_vals[x] <= self.cat_num]
        #self.num_sensors = [x for x in list(set(self.ids)-set(self.timeSensors)) if unique_vals[x] > self.cat_num]

        
        
        
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.num_sensors),
                ('cat', self.categorical_transformer, self.cat_sensors)])
        
        self.preprocessor.fit(self.full_traindf)
        
        self.train_transformed = [] # List of transformed training data.
        for df in self.train_raw:
            trans = self.preprocessor.transform(df) #transformed training data
            try:
                self.train_transformed.append(pd.DataFrame(trans.todense()))
            except:
                self.train_transformed.append(pd.DataFrame(trans))
        #self.train_transformed['day_time_x'] = np.cos(self.train_raw['time']%(24*60*60*1000)*(2*np.pi/(24*60*60*1000)))
        #self.train_transformed['day_time_y'] = np.sin(self.train_raw['time']%(24*60*60*1000)*(2*np.pi/(24*60*60*1000)))
#         self.train_transformed['time'] = self.train_raw['time']

        #Now make dictionaries from the new variable names to the old and back
        
        # First sensor name to id list
        # First the numerical sensors
        sen_to_idx_list = [] # Will make a dictionary out of this
        for i in range(len(self.num_sensors)):
            transCols = list(self.train_transformed[0].columns)
            sen_to_idx_list = sen_to_idx_list + [(self.num_sensors[i],[transCols[i]])]
        # Now the categorical sensors
        base = len(self.num_sensors) # Where numbering of one-hot encoding of catagorical sensors starts.
        for x in self.cat_sensors:
            newpair = (x, list(range(base,base+self.full_traindf[x].nunique())))
            sen_to_idx_list.append(newpair)
            base+= self.full_traindf[x].nunique()
        # The dictionary mapping sensor id to the list of indices in the transformed data that represent it
        self.sen_to_idx_dict = dict(sen_to_idx_list)
        
        #Now use the dictionary to get a list of the ids for the non-categorical and categorical sensors
        self.num_id_list = [x[0] for x in self.sen_to_idx_dict.values() if len(x) ==1]
        self.cat_id_list = [x[0] for x in self.sen_to_idx_dict.values() if len(x) > 1]
        
        
        # Next making the reverse dictionary from index to snesor id
        idx_to_sen_list = []
        for pair in sen_to_idx_list:
            sublist = [(idx, pair[0]) for idx in pair[1]]
            idx_to_sen_list = idx_to_sen_list + sublist
        self.idx_to_sen_dict = dict(idx_to_sen_list)
        
        # Now make a data frame to hold aggregate values for each numerical sensor (R^2, etc)
        self.agg_df = pd.DataFrame(index=self.num_sensors)
        
        self.train_gens = [] # List of training generators, one for each training set.
        for df in self.train_transformed:
            print(df.iloc[0])
            train_gen = TimeseriesGenerator(df.values,
                                        df[self.num_id_list].values,
                                        length=self.windowL, sampling_rate=1,stride=1,
                                        batch_size=self.batch_size)
            self.train_gens.append(train_gen)
        
        # Finally make the Keras sequence generators
        # Make the Keras sequence generator for the training data
#         train_n = int(len(self.train_transformed)*(1-train_val_ratio))
#         val_n = len(self.train_transformed) - train_n
#         self.train_gen = TimeseriesGenerator(self.train_transformed.iloc[:train_n,:].values,
#                                         self.train_transformed[self.num_id_list].iloc[:train_n,:].values,
#                                         length=self.windowL, sampling_rate=1,stride=1,
#                                         batch_size=self.batch_size)
        # Make the Keras sequence generator for training time validation data
 #       self.train_val_gen = TimeseriesGenerator(self.train_transformed.iloc[train_n:,:].values,
 #                                       self.train_transformed[self.num_id_list].iloc[train_n:,:].values,
 #                                       length=self.windowL, sampling_rate=1,stride=1,
 #                                       batch_size=self.batch_size)

        
        
    def preprocess(self, data):
        data_index = data.index.tolist()
        
        
        # Difference the sensors highly correlated with time
#         timedf = pd.DataFrame(index=data.index)
#         timedf['time'] = data.time
#         for sensor in self.timeSensors:
#             if sensor == 'time':
#                 continue
#             parts = re.split(r'\_\_',sensor) #JCSAT NAMING CONVENTIONS
# #             parts = re.split(r':',sensor) #AM10 NAMING CONVENTIONS
#             diffname = parts[0]+'_DIFF__'+parts[1]
#             timedf[diffname]=data[sensor].diff()
#         timedf.fillna(method='bfill', inplace=True)
#         # Drop diff'ed sensors correlated with time (and drop time)
#         timedf.drop(self.difftimeSensors, axis=1, inplace=True)

#         frames = [data[list(set(self.ids)-set(self.timeSensors))], timedf]
#         fittingdf = pd.concat(frames, axis=1)
        trans = self.preprocessor.transform(data.drop(['time'], axis=1)) #transformed training data
        try:
            data_transformed = pd.DataFrame(trans.todense(),index=self.test_raw.index)#,index=self.test_raw.index)
        except:
            data_transformed = pd.DataFrame(trans)#,index=self.test_raw.index
        data_transformed['time'] = list(data['time'])
        #data_transformed['day_time_x'] = np.cos(data_transformed['time']%(24*60*60*1000)*(2*np.pi/(24*60*60*1000)))
        #data_transformed['day_time_y'] = np.sin(data_transformed['time']%(24*60*60*1000)*(2*np.pi/(24*60*60*1000)))
        data_transformed.drop(['time'],axis=1, inplace=True)
        return data_transformed

    
    #try:
    #            self.train_transformed.append(pd.DataFrame(trans.todense()))
    #        except:
    #            self.train_transformed.append(pd.DataFrame(trans))
    
    
    
    def process_test(self, test_raw):
        self.oldPred = True
        self.test_raw = test_raw
        self.test_transformed = self.preprocess(test_raw)
        
        # Make the Keras sequence generator for the test data
        self.test_gen = TimeseriesGenerator(self.test_transformed.values,
                                            self.test_transformed[self.num_id_list].values,
                                            length=self.windowL, sampling_rate=1,stride=1,
                                            batch_size=1)
        
    def process_validation(self):
        self.validate_transformed = self.preprocess(self.validate_raw)
        
        # Make the Keras sequence generator for the validate data
        self.val_gen = TimeseriesGenerator(self.validate_transformed.values,
                                           self.validate_transformed[self.num_id_list].values,
                                           length=self.windowL, sampling_rate=1,stride=1,
                                           batch_size=1)

    def make_model(self,
                   ave_pool = 1,
                   pre_dense = True,
                   pre_dense1_units = 25,
                   pre_dense2_units = 10,
                   pre_drop = 0,
                   kernel_reg = None,
                   recurrent_reg = None,
                   lstm_dim = 1000,
                   drop = 0.2,
                   metric='mean_squared_error'):
        """
        Make the lstm model.
        pre_dense = whether or not to have an initial dense layer before entering the recurrent cell.
        pre_dense_units = output size of initial dense layer
        pre_drop = dropout rate applied as a layer before entering the recurrent cell.
        
        """
        self.lstm_dim = lstm_dim
        #print('STARTING')
        
        
        
        sensN = len(self.train_transformed[0].columns)  # number of sensors (eliminating the two time ones)
        outN = len(self.num_id_list) # number of output sensors; the non-categorical ones        
        
        input_layer = Input(shape=(self.windowL, sensN), dtype='float32', name='input')
        memory_layer = LSTM(self.lstm_dim, return_sequences=True)(input_layer)
        memory_layer = LSTM (int(self.lstm_dim/2), return_sequences=False)(memory_layer)
        repeated = RepeatVector(self.windowL)(memory_layer)
        memory_layer = LSTM (int(self.lstm_dim/2), return_sequences=True)(repeated)
        memory_layer = LSTM (self.lstm_dim,  return_sequences=True)(memory_layer)
        decoded_inputs = TimeDistributed(Dense(units=outN, activation='linear'))( memory_layer)
        
        #  Try spatial dropout?
        dropout_input = Dropout(drop)(input_layer)
        concat_layer = concatenate([dropout_input, decoded_inputs])
        #memory_layer = LSTM (units=self.lstm_dim, return_sequences=False)(concat_layer)
        memory_layer = LSTM (units=self.lstm_dim, kernel_regularizer = kernel_reg, recurrent_regularizer = recurrent_reg, return_sequences=False)(concat_layer)
        preds = Dense(units=outN, activation='linear')(memory_layer)
        
        self.model = Model(input_layer, preds)
        #self.model.compile(optimizer='adagrad', loss='mean_squared_error')
        self.model.compile(optimizer='adam', loss='mean_squared_error')             
        
        print(self.model.summary())
       
            

    def fit_model(self, fullepochs = 5, verbose=True, early_stop = False, patience = 4):
        self.oldPred = True
        self.mean_train_loss = [np.inf]*patience
        self.mean_val_loss = [np.inf]*patience
        if early_stop:
            callbacks = [EarlyStopping(monitor='val_loss', patience=patience)]
        else:
            callbacks = None
            
        minloss = np.inf
        for i in range(fullepochs):
            print('Full Epoch {}'.format(i+1))
            curr_train_loss = []
            curr_val_loss = []
            for generator in self.train_gens:
                print(generator)
                self.model.fit_generator(generator,
                           validation_data = self.test_gen,
                           epochs = 1,
                           callbacks=callbacks,
                           verbose=verbose,
                           shuffle=True)
                curr_train_loss.append(self.model.history.history['loss'])
                curr_val_loss.append(self.model.history.history['val_loss'])
                                     
                self.train_loss = self.train_loss + self.model.history.history['loss']
                self.val_loss = self.val_loss + self.model.history.history['val_loss']
            #curr_train_mean = np.mean(curr_train_loss)
            #current_val_mean = np.mean(curr_val_loss)
            self.mean_train_loss.append(np.mean(curr_train_loss))
            self.mean_val_loss.append(np.mean(curr_val_loss))
            print('Full epoch val_loss = {}'.format(np.mean(curr_val_loss)))
            if min(self.mean_val_loss[-patience:]) <= minloss:
                minloss = min(self.mean_val_loss)
            else:
                break
                

        
    def predict_train(self):
        """
        Predict on the training set for the purpose of determining which sensors are well-predictable.
        """
        self.train_pred = True
        self.train_preds = self.model.predict_generator(self.train_gen)
        # Because the validation set is taken from the training set before scaling and training, this must be cut.
        self.train_y = self.train_transformed[self.num_id_list].values[self.windowL:len(self.train_preds)+self.windowL]

        self.train_error_var = np.var(self.train_y - self.train_preds, axis=0) # variance of the predictions error on training
        self.train_error_mean = np.mean(self.train_y - self.train_preds, axis=0) # mean of the prediction error on training
        self.train_abserrordf = pd.DataFrame(np.abs(self.train_y - self.train_preds), columns=self.num_id_list)
        self.train_abs_errordf = pd.DataFrame(np.abs(self.train_y - self.train_preds), columns=self.num_id_list)
        self.train_errordf = pd.DataFrame((self.train_y - self.train_preds), columns=self.num_id_list)
        
        
    def predict_validate(self):
        """
        Predict on the validation set for the purpose of determining which sensors are well-predictable.
        Actually, for GPS3, we use training data for this purpose because the time period is short.
        """
        self.validate_pred = True
        self.validate_preds = []
        self.validate_y = []
        self.r2 = []
        self.aggdflist = [pd.DataFrame(index = self.num_sensors)]*len(self.train_raw)
        self.validate_abs_errordflist = []
        self.validate_errordflist = []
        for i in range(len(self.train_raw)):
            self.validate_preds.append(self.model.predict_generator(self.train_gens[i]))
            self.validate_y.append(self.train_transformed[i][self.num_id_list].values[self.windowL:len(self.validate_preds[i])+self.windowL])
            self.r2.append(r2_score(self.validate_y[i], self.validate_preds[i], multioutput='raw_values'))
            self.aggdflist[i]['validate_r2'] = self.r2[i]
            validate_error_var = np.var(self.validate_y[i] - self.validate_preds[i], axis=0) # variance of the predictions error on training
            self.aggdflist[i]['validate_error_var'] = validate_error_var
            validate_error_mean = np.mean(self.validate_y[i] - self.validate_preds[i], axis=0) # mean of the prediction error on training
            self.aggdflist[i]['validate_error_mean'] = validate_error_mean
#        self.validate_abserrordf = pd.DataFrame(np.abs(self.validate_y - self.validate_preds), columns=self.num_id_list)
            self.validate_abs_errordflist.append(pd.DataFrame(np.abs(self.validate_y[i] - self.validate_preds[i]), columns=self.num_id_list))
            self.validate_errordflist.append(pd.DataFrame((self.validate_y[i] - self.validate_preds[i]), columns=self.num_id_list))
        
        self.agg_df = pd.DataFrame(index = self.num_sensors)
        self.agg_df['validate_r2'] = sum(df['validate_r2'] for df in self.aggdflist)/len(self.aggdflist)
        self.agg_df['validate_error_var'] = sum(df['validate_error_var'] for df in self.aggdflist)/len(self.aggdflist)
        self.agg_df['validate_error_mean'] = sum(df['validate_error_mean'] for df in self.aggdflist)/len(self.aggdflist)
        
        
     
     
         
       
    def predict_new(self):
        self.oldPred = False
        self.preds = self.model.predict_generator(self.test_gen)
        self.test_y = self.test_transformed[self.num_id_list].values[self.windowL:]
        self.test_abs_errordf = pd.DataFrame(np.abs(self.test_y - self.preds), columns=self.num_id_list)
        self.test_errordf = pd.DataFrame((self.test_y - self.preds), columns=self.num_id_list)
        
          

            
    def score_test(self, r2_threshold = -np.inf):
        self.r2_threshold = r2_threshold # # R^2 cutoff for using a sensor in score, etc.
#         self.high_quality_sensors2 = [x for (i,x) in enumerate(self.num_id_list) if self.r2[i] >= self.r2_threshold]
        self.high_quality_sensor_names = [x for x in self.agg_df.index.tolist() if self.agg_df.loc[x,'validate_r2'] >= self.r2_threshold]
        self.high_quality_sensors = [self.sen_to_idx_dict[x][0] for x in self.high_quality_sensor_names]
        self.high_qualitypred_df = pd.DataFrame(self.preds, columns=self.num_id_list)[self.high_quality_sensors]
        if self.oldPred:
            print('Predictions are old, so re-predicting.')
            self.predict_new()
        high_quality_y = pd.DataFrame(self.test_y, columns = self.num_id_list)[self.high_quality_sensors]
        self.score = np.linalg.norm(self.high_qualitypred_df.values - high_quality_y.values,axis=1) #score not scaled by R^2
        
        # For R^2 scaling for the score
        
        error_mat = np.abs(self.high_qualitypred_df.values - high_quality_y.values)
        scaled_error_mat = error_mat * self.agg_df['validate_r2'].values[np.where(self.agg_df['validate_r2'].values > self.r2_threshold)].reshape(1,-1)
        self.r2_score = np.sum(scaled_error_mat, axis=1)
        
    def score_validate(self):
        """
        Again, the validation set used for the purpose of determing quality sensors is the training data.
        """
        
        # These are lists of the associated quantities for each training set.  The vague plan is to average them to find
        # overall measured durign training.
        high_qualitypred_dflist = []
        high_quality_ylist = []
        valid_scorelist = [] # List of anomaly scores during the validation sets (which, for this is training).
        for i in range(len(self.train_raw)):
            high_qualitypred_dflist.append(pd.DataFrame(self.validate_preds[i], columns=self.num_id_list)[self.high_quality_sensors])
            high_quality_ylist.append(pd.DataFrame(self.validate_y[i], columns = self.num_id_list)[self.high_quality_sensors])
            valid_scorelist.append(np.linalg.norm(high_qualitypred_dflist[i].values - high_quality_ylist[i].values,axis=1)) #score not scaled by R^2
        self.valid_score = sum(valid_scorelist)/len(self.train_raw)
        
        # For R^2 scaling for the score
        error_matlist = []
        scaled_error_matlist = []
        
        for i in range(len(self.train_raw)):
            error_matlist.append(np.abs(high_qualitypred_dflist[i].values - high_quality_ylist[i].values))
            scaled_error_matlist.append(error_matlist[i] * self.agg_df['validate_r2'].values[np.where(self.agg_df['validate_r2'].values > self.r2_threshold)].reshape(1,-1))
        
        scaled_error_mat = sum(scaled_error_matlist)/len(self.train_raw)
        
        self.r2_valid_score = np.sum(scaled_error_mat, axis=1)
          
       