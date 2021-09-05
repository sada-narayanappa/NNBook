#!/usr/local/bin/python 

import os, datetime, glob, sys
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.append(".")
sys.path.append("gen")
import time, datetime

'''
The following code: it is just here to protect my API-KEY code not really useful
for the problem at hand. If your key is in your or update your API_KEY then you may safely ignore 
this code.
'''

def getkey(key='password'):
    API_KEY, lines, file =None, None, os.path.expanduser('~/.keys/keys.json')
    if os.path.exists(file):
        with open(file, 'r') as f:
            r = f.read()
        j = eval(r)
        return j['AV_API_KEY'], j['NEWSAPI_KEY']

    #alpha_vantage key
    avk = decrypt('baTEje52rx+kAsuAN9PdxMeC03p/HuRVTzskLiso1/c=', key)
    newsk = decrypt('505Db5sDvHvptBPzE8IhsewneuanOKV3gKpN+26lS3A=', key)
    return avk, newsk;
        
def encrypt(msg_text = b'message', secret_key='password'):
    if (type(msg_text) == str):
        msg_text = bytes(msg_text, encoding='utf-8').rjust(32)
    if (type(secret_key) == str):
        secret_key = bytes(secret_key, encoding='utf-8') .rjust(32)

    cipher = AES.new(secret_key,AES.MODE_ECB) 
    encoded = base64.b64encode(cipher.encrypt(msg_text))
    ret = encoded.decode("utf-8")
    print(ret)
    return ret

def decrypt(encoded, secret_key='password'):
    if (type(secret_key) == str):
        secret_key = bytes(secret_key, encoding='utf-8') .rjust(32)

    cipher = AES.new(secret_key,AES.MODE_ECB) 
    if (type(encoded) == str):
        encoded = bytes(encoded, encoding='utf-8')
    decoded = cipher.decrypt(base64.b64decode(encoded))
    ret =decoded.decode("utf-8").strip()
    print(ret)
    return ret

'''
This will read required symbols and saves them to data directory
'''
def save_data(symbol, API_KEY="", check=True):
    from alpha_vantage.timeseries import TimeSeries
    
    outf = f'daily_{symbol}.csv'
    if (check and os.path.exists(outf)):
        
        dt = datetime.datetime.fromtimestamp(os.path.getmtime(outf))
        dn = datetime.datetime.now()
        ts = (dn - dt)
        hr = (ts.days * 24 * 60 * 60 + ts.seconds)//60//60
        if (hr < 4): #if it was created less than 4 hours ago
            print(f"{outf:22} exists, ... recently crested < 4 at {dt} ")
            return;
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    
    retry = 0
    data = None
    while retry <= 5:
        try:
            print(f"Getting data for {symbol}")
            data, meta_data = ts.get_daily(symbol, outputsize='full')
            break;
        except ValueError as ve:
            retry += 1
            print(f"Sleep for a Minute, {retry}/5 attempts");
            time.sleep(60)
            
    if data is None:
        print(f"Could not get data for {symbol}")
        return data
    
    data.insert(0, 'timestamp', value=data.index)

    data.columns = 'timestamp,open,high,low,close,volume'.split(',')
    data.to_csv(outf, index=False)
    return data

'''
Read all the files in data with daily_*, reads them and returns a list
'''
def read_data():
    a={}
    for f in glob.glob('daily_*'):
        symbol = os.path.basename(os.path.splitext(f)[0]).split("_")[1]
        print(f'Reading {f} Symbol: {symbol}')
        df = pd.read_csv(f)
        df.sort_values(by='timestamp', ascending=False, inplace=True)
        df.index=(range(0,len(df)))
        ncs = ['timestamp'] + [f'{symbol}_{c}' for c in df.columns[1:]]
        df.columns = ncs
        a[symbol] = df
        
    minrows = min([len(d) for d in a.values()])
    return a, minrows, df

'''
Combines all the dataframes into one

The problem for us to join multiple index funds from ASIA is that they have different holidays
Therefore we have gaps in the trading days. Therefore 
'''
def combine_data(a, outp="stockdata.csv"):
    # Different excahnges have various holidays, therefore values may be missing  
    # Get All Data Frames and their corresponding time stamps
    #
    ar=np.array([d['timestamp'].values for d in a.values()])
    at=np.concatenate(ar)
    at=set(at)
    af = pd.DataFrame()
    af['timestamp'] = list(at);

    for k,v in a.items():
        #print(f"Getting {k:32} \r", end='')
        af=pd.merge(af,v, how="left", left_on="timestamp".split(), right_on="timestamp".split())
    print()
    af.sort_values(by='timestamp', ascending=True, inplace=True)
    af.dropna(inplace=True)
    af = af.reset_index(drop=True)
    #af = af.fillna(method='ffill' ).fillna(method='bfill')
    af.to_csv(outp, index=False)
    return af

def addDummyCols(df):
    tf1=df
    if ( "MSFT_+ve" not in tf1.columns):
        tf1.insert(1, "MSFT_+ve", value=[f"S__{k}" for k in np.random.randint(0,3, size=len(tf1))] )
    if ( "MSFT_-ve" not in tf1.columns):
        #tf.insert(1, "MSFT_-ve", value=[f"S__{k}" for k in np.random.randint(0,1, size=len(nf))] )
        tf1.insert(1, "MSFT_-ve", value=[k for k in np.random.randint(0,2, size=len(tf1))] )
    tf1.to_csv("stockdata_ext.csv", index=False);
    return tf1


def getdata(symbs='MSFT GLD GOOGL SPX AAPL IBM' , dontforce=False):
    
    stockfile="data/stockdata.csv"
    if (dontforce and os.path.exists(stockfile)):
        print(f"{stockfile} exists")
        return
    
    API_KEY = None  # Put your API KEY if you need to test data download or just use the data
    API_KEY, NEWS_API_KEY = API_KEY or getkey()

    #print("API_KEY ", API_KEY)
    for f in symbs.split():
        save_data(f, API_KEY=API_KEY)

    ASIA='''TOKYO 6758.T HITACHI 6501.T HNGKNG 0168.HK SHANGAI 601288.SS SHNZEN'''
    s=ASIA.split()
    ASIA = {k[0]:k[1] for k in   zip(s[0::2], s[1::2])}

    for k, v in ASIA.items():
        print(f'getting data for {k} => symbol {v}')
        save_data(v, API_KEY)

def getCombined():
    #Get All data togehe
    a, maxrows, ldf  = read_data()
    nf= combine_data(a, "stockdata.csv")
    addDummyCols(nf)
    return nf
    
#-----------------------------------------------------------------------------------
#*** NOTE: DO NOT EDIT THIS FILE - THIS iS CREATED FROM: inv_utils.ipynb
def inJupyter():
    try:    get_ipython; return True
    except: return False
#-----------------------------------------------------------------------------------
if __name__ == '__main__':
    if (not inJupyter()):
        t1 = datetime.datetime.now()
        getdata()
        getCombined();
        t2 = datetime.datetime.now()
        print(f"All Done in {str(t2-t1)} ***")
    else:
        pass
        '''
        a, minrows, ldf  = read_data()
        stockfile="data/stockdata.csv"
        nf= combine_data(a, stockfile)
        addDummyCols(nf)
        '''
