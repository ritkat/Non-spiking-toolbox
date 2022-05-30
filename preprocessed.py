from pickle import TRUE
import numpy as np
import pandas as pd
import time
import statsmodels.api as sm
from scipy import signal
from sklearn.preprocessing import StandardScaler
import EEGExtract as eeg
import pywt
from scipy.signal import hilbert, chirp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
# Import the RFE from sklearn library
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from genetic_selection import GeneticSelectionCV
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import os
import pyeeg
import nolds
import copy
from pywt import wavedec
from tqdm import tqdm
from tqdm.notebook import tqdm
from args_final import args as my_args
from utils import *

args=my_args()

def preprocess(data_train, data_test,f_split, fs, c_ref):

  data_2_subs=data_train  

  if(c_ref==True):
  #Common Average Reference
    for j in range(0, data_train.shape[0]):
        car=np.zeros((data_2_subs.shape[2],))
        for i in range(0, data_train.shape[1]):
            car= car + data_2_subs[j,i,:]
        car=car/data_train.shape[1]
        #car.shape
        for k in range(0, data_train.shape[1]):
            data_2_subs[j,k,:]=data_2_subs[j,k,:]-car
  #subsampling by 4 
  data_2_subs_t=data_test
  '''data_2_subs_t=np.zeros((data_test.shape[0], data_test.shape[1], int(data_test.shape[2]/4)))
  for i in range(0, data_test.shape[0]):
      for j in range(0, data_test.shape[1]):
          data_2_subs_t[i, j, :]=signal.resample(data_2_sub_t[i, j, :], int(data_test.shape[2]/4))'''

  #data_2_subs_t.shape
  #Common Average Reference
  if(c_ref==True):
    for j in range(0, data_2_subs_t.shape[0]):
        car=np.zeros((data_2_subs_t.shape[2],))
        for i in range(0, data_2_subs_t.shape[1]):
            car= car + data_2_subs_t[j,i,:]
        car=car/data_2_subs_t.shape[1]
        #car.shape
        for k in range(0, data_2_subs_t.shape[1]):
            data_2_subs_t[j,k,:]=data_2_subs_t[j,k,:]-car

  #Standard Scaler

  '''for j in range(0, data_train.shape[0]):
      kr=data_2_subs[j,:,:]
      kr=data_2_subs[j,:,:]
      
      scaler=StandardScaler().fit(kr.T)
      data_2_subs[j,:,:]=scaler.transform(kr.T).T'''
      
  #scaler = StandardScaler()
  #param_ls=[]
  mu_l={}
  std_l={}
  for j in range(data_2_subs.shape[1]):
    mu_l[str(j)]=[]
    std_l[str(j)]=[]
    
  for i in range(data_2_subs.shape[0]):
    for j in range(data_2_subs.shape[1]):
      mu=np.mean(data_2_subs[i,j,:])
      std=np.std(data_2_subs[i,j,:])
      mu_l[str(j)].append(mu)
      std_l[str(j)].append(std)
  
  for j in range(data_2_subs.shape[1]):
    mu_l[str(j)]=sum(mu_l[str(j)])/len(mu_l[str(j)])
    std_l[str(j)]=sum(std_l[str(j)])/len(std_l[str(j)])
    
  for i in range(data_2_subs.shape[0]):
    for j in range(data_2_subs.shape[1]):
      data_2_subs[i,j,:]=(data_2_subs[i,j,:]-mu_l[str(j)])/std_l[str(j)]
      
  for i in range(data_2_subs_t.shape[0]):
    for j in range(data_2_subs_t.shape[1]):
      data_2_subs_t[i,j,:]=(data_2_subs_t[i,j,:]-mu_l[str(j)])/std_l[str(j)]
      
  data_hilbert=copy.deepcopy(data_2_subs)
  data_hilbert_t=copy.deepcopy(data_2_subs_t)
  
  for j in range(0, data_hilbert.shape[0]):
    for i in range(0, data_hilbert.shape[1]):
      data_hilbert[j,i,:]=np.imag(hilbert(data_hilbert[j,i,:]))
  
  for j in range(0, data_hilbert_t.shape[0]):
    for i in range(0, data_hilbert_t.shape[1]):
      data_hilbert_t[j,i,:]=np.imag(hilbert(data_hilbert_t[j,i,:]))
  
  return  data_2_subs, data_2_subs_t, data_hilbert, data_hilbert_t