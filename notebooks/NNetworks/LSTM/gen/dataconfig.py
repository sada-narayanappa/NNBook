#!/usr/local/bin/python 

import re, sys, os, datetime, getopt, glob, argparse, datetime, json, base64, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
sys.path.append(".")
sys.path.append("gen")
import utils1

#~~~~ Find any sensor highly correlated with time and drop them.
def detectTimeCorrelated(df, val=0.94):
    timeCorSensors = []
    if ( not 'time' in df.columns): #assume first column is time column
        dcols = ['time'] + [c for c in df.columns[1:]]
        df.columns = dcols
    
    timeser = pd.Series(df[['time']].values.reshape(-1))
    if ( timeser.dtype != np.number ):
        timeser = pd.to_datetime(timeser).astype(int)
    
    
    DROP_INDEX = 0;
    for sensor in df.columns:
        if (sensor == 'time'):
            continue;
        #print(f"#Testing {sensor}...")
        sensorSeries = pd.Series(df[sensor].values.reshape(-1))
        for i in range(8):
            c1 = timeser[i:].corr(sensorSeries[i:])
            c2 = timeser[i:].corr(sensorSeries[:-i])
            if np.abs(c1) >= val or np.abs(c2) >= val:
                timeCorSensors.append(sensor)
                DROP_INDEX = max(DROP_INDEX, i) #lets drop first few rows
                break;
                
    if ( len(timeCorSensors) > 0):
        print(f"#Time Cor: #{len(timeCorSensors)}, #Shape before:{df.shape}")
        #df.drop(timeCorSensors, axis=1, inplace=True)
        #df = df[DROP_INDEX:]
        #print(f"#After dropping: {DROP_INDEX} =>{df.shape}")
        
    return timeCorSensors
#-----------------------------------------------------------------------------------
def precheck(df):
    cols = df.columns[df.dtypes.eq('object')]
    if (len(cols) > 0):
        print(f"WARNING: *** Non numeric columns => {cols}")
        return 0
    return 1
#-----------------------------------------------------------------------------------
'Covert to one_hot encoding with prefix for columns'
def makeOneHotCols(tf1, oheCols=[]):
    ret = []
    for c in oheCols:
        one_hot = pd.get_dummies(tf1[c])
        ret += [f'{c}___{k}' for k in one_hot.columns]

    return ret
#-----------------------------------------------------------------------------------
def detectCols(file, nUnique=4, tcoeff=0.92):
    tf1 = file
    if (type(tf1) == str):
        tf1 = pd.read_csv(tf1, comment="#")
    
    #Lets check if it has any non-numeric columns! Warning
    precheck(tf1)
    
    unique_vals  = tf1.nunique()
    constantCols = unique_vals[ unique_vals == 1].index                             # constant Columns
    onehotECols  = unique_vals[(unique_vals > 2 ) & (unique_vals<=nUnique)].index   # Categorical Columns
    categorCols  = unique_vals[(unique_vals >=2 ) & (unique_vals <= nUnique)].index # Categorical Columns
    binaryCols   = unique_vals[(unique_vals == 2)].index                      # Binary

    numericCols  = tf1.select_dtypes(include=np.number).columns           # numerics
    numericCols  = [c for c in numericCols if c not in categorCols]
    numericCols  = [c for c in numericCols if c not in constantCols]
    notNumerics  = tf1.select_dtypes(exclude=np.number).columns           # non - numerics
    notNumerics  = [c for c in notNumerics if c not in categorCols]       # non - numerics

    try:
        timeCorCols  = detectTimeCorrelated(tf1, tcoeff)
    except:
        timeCorCols = []
    
    onehotEC_ext = makeOneHotCols(tf1, onehotECols)
   
    ret1 =f'''[START]
{{
    "file"           : {[file] if (type(file) == str) else ["??"]},
    "nrowsXncols"    : {[len(tf1), len(tf1.columns )] }     , 
    "number_Unique"  : {nUnique}            , 
    "constantCols"   : {list(constantCols )},   # No Signals
    "#constantCols"  : {len(constantCols  )},   # No Signals
    "categorCols"    : {list(categorCols  )},   # Categorical Columns
    "#categorCols"   : {len(categorCols   )},   # Categorical Columns
    "onehotECols"    : {list(onehotECols  )},   # Cats > 2 and < Unique Values
    "onehotEC_ext"   : {list(onehotEC_ext )},   # Cats > 2 and < Unique Values
    "#onehotECols"   : {len(onehotECols   )},   # Cats > 2 and < Unique Values
    "binaryCols"     : {list(binaryCols   )},   # Binary
    "#binaryCols"    : {len(binaryCols    )},   # Binary
    "notNumerics"    : {list(notNumerics  )},
    "timeCorrelation": {tcoeff             },   # Time correlated
    "timeCorrCols"   : {list(timeCorCols  )},   # Time correlated Columns
    "#timeCorrCols"  : {len(timeCorCols   )},    # Time correlated Columns
    "excludePattern" : [] , #Exclude patterns
    "includePattern" : [] , #include patterns
    "dropColumns"    : [],
    "diff_suffix"    : {['__diff1']},
    "addDiffs"       : [],
    "train_pct"      : .9,
    "#numericCols"   : {len(numericCols   )},  
    "scaleInputs"    : {list(numericCols  )},  
    "scaleOutputs"   : {["$scaleInputs"]},  
    "inputs"         : {["$binaryCols", "$scaleInputs", "$onehotECols"]},
    "outputs"        : {["$scaleOutputs"]},
#-----Copy this generated file and add customization
    "tsParams"       : {{"length": 50, "batch_size": 1, "stride": 1, "sampling_rate": 1}},
    "networkModel"   : ["NN1 -> Example"],
    "lookahead"      : 1,
    "nsteps"         : 1,
    "scale"          : 1,
    "scaler"         : ["sklearn.preprocessing.MinMaxScaler()"],
    "scaler"         : ["sklearn.preprocessing.StandardScaler()"],
    "scalerXString"  : [],
    "scalerYString"  : []
}}
[END]
    '''

    return ret1, tf1;
#-----------------------------------------------------------------------------------
#*** NOTE: DO NOT EDIT THIS FILE - THIS iS CREATED FROM: inv_utils.ipynb
def inJupyter():
    try:    get_ipython; return True
    except: return False
    
def process():
    n  = len(sysargs.input_files)
    un = sysargs.unique
    tc = sysargs.tcoeff
    for i, file1 in enumerate(sysargs.input_files):
        print(f"#=>Processing {i+1}/{n} {file1} #unique: {un} tcoeff: {tc} - standby")
        outs, df = detectCols(file1, un, tc)
        
        break;
    print(outs)
    return outs
    
#-----------------------------------------------------------------------------------
sysargs=None
def addargs():
    sysargs = None
    p = argparse.ArgumentParser(f"{os.path.basename(sys.argv[0])}:")
    p.add_argument('-u', '--unique', type=int,   default=6,    help="# of unique values!")
    p.add_argument('-t', '--tcoeff', type=float, default=0.94, help="# Time Correlation value Sensors!")
    p.add_argument('args', nargs=argparse.REMAINDER)
    p.add_argument('input_files',action="store", type=str, nargs='+', help="input file(s)")

    #p.print_help() # always print -help
    try:
        sysargs=p.parse_args(sys.argv[1:])
        #print(f'using:\n{sysargs}')
    except argparse.ArgumentError as exc:
        #par.print_help()
        print(exc.message )
        
    return sysargs
#-----------------------------------------------------------------------------------
if __name__ == '__main__':
    if (not inJupyter()):
        t1 = datetime.datetime.now()
        sysargs = addargs()
        process()
        t2 = datetime.datetime.now()
        print(f"#All Done in {str(t2-t1)} ***")
    else:
        pass
