# matplotlib inline
import matplotlib.pyplot as plt
from numpy import *
from collections import Counter
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, cluster, preprocessing, decomposition
from sklearn.decomposition import PCA
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy.random as random
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import time
from IPython.display import display
from IPython.display import Image

from scipy	import stats;

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


# -*- coding: utf-8 -*-
# t- test
def ComputeDegreesOfFreedomFor_t_test(x,y, equal_variance = True):
    n1 = len(x)
    n2 = len(y)
    if equal_variance:
        return n1 + n2 -2;
    s1 = x.var()
    s2 = y.var()
    num = pow((s1/n1 + s2/n2),2)
    den = (pow(s1/n1,2) /(n1 -1)) + (pow(s2/n2,2) /(n2 -1))
    ret = num/den;
    ret = round(ret)
    return ret;

def t_critical(alpha, df, tails=1):
    t_critical_one_tailed=stats.t.ppf(1-alpha, df);
    t_critical_two_tailed=stats.t.ppf(1-alpha/2, df);

    if ( tails == 1):
        return t_critical_one_tailed;
    else:
        return t_critical_two_tailed;
#       
# Returns equal Variance = true if there is no difference in variances
#
#
def Ftest(x,y, alpha = 0.05):
    F = x.var()/ y.var();
    n1 = len(x);  # degrees of freedom
    n2 = len(y);  # degrees of freedom
    cl = 1 - alpha; # Confidence Level
    F_c = stats.f.ppf(cl, n1 -1 , n2 - 1);  # n1-1, n2-1 are degrees of freedom
    p = 1- stats.f.cdf(F, n1 - 1, n2 - 1) 

    equal_variance = True;
    if ( F > F_c ):
        equal_variance = False;

    ostr =  "Equal Variance: " + str(equal_variance) + \
            ", F= "+ str(F) + ", F_critical="+ str(F_c) + ",P=" + str(p);
    return (equal_variance , F, F_c, p, ostr);

#
# Chi square Critical value 
def Chi2Critical(df, alpha = 0.05):
    cl = 1 - alpha; # Confidence Level
    Chi2_c = stats.chi2.ppf(cl, df);
    return Chi2_c;
    
    
    
    
