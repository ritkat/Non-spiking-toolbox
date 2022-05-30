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

args=my_args()

def autocorr(x):
  result = np.correlate(x, x, mode='full')
  return result[result.size//2:]

def mean1(expo):
  N=len(expo)
  sum=0
  for i in range(N):
    if i <= 0.75*N and i >= 0.25*N:
        w = 1
    else:
      w = 0.5
    sum=sum+w*np.absolute(expo[i])
  f=sum/N
  return f

def mean2(expo):
  N=len(expo)
  sum=0
  for i in range(N):
    if i <= 0.75*N and i >= 0.25*N:
      w = 1
    elif i < 0.25*N :
      w = 4*i/N
    else:
      w = 4*(N-i)/N
    sum=sum+w*np.absolute(expo[i])
  f=sum/N
  return f

def log_detec(expo):
  N=len(expo)
  sum=0
  for i in range(N):
    if(expo[i]==0):
      continue
    sum=sum+np.log(np.absolute(expo[i]))
  f=sum/N
  return np.exp(f)

def abs_diff(expo):
  N=len(expo)
  sum=0
  for i in range(N-1):
    sum=sum+np.absolute(expo[i+1]-expo[i])
  f=sum/(N-1)
  return f

def mean_freq(expo):
  f, Pxx_den = signal.periodogram(expo, fs=1000)
  sum=0
  for i in range(Pxx_den.shape[0]):
    sum=sum+(f[i]*Pxx_den[i])
  ret=sum/np.sum(Pxx_den)
  return ret

def freq_atmax(expo):
  f, Pxx_den = signal.periodogram(expo, fs=1000)
  sum=0
  ind=np.where(Pxx_den==np.amax(Pxx_den))[0]
  ret=f[ind]
  return ret[0]

def max_psd(expo):
  f, Pxx_den = signal.periodogram(expo, fs=1000)
  ret=np.amax(Pxx_den)

  return ret



def segment(data_trial, segment_length=500):
  data_final=np.array([])
  for i in range(0, data_trial.shape[0]):
    data_temp=data_trial[i,:,:]
    data_temp2=np.array([])
    for j in range(int(data_temp.shape[1]/segment_length)):
      llim=j*segment_length
      data_temp1=data_temp[:,llim:llim+segment_length]
      if j==0:
        data_temp2=data_temp1[np.newaxis,:,:]
      else:
        data_temp2=np.vstack((data_temp2, data_temp1[np.newaxis,:,:]))

    if i==0:
      data_final=data_temp2

    else:
      data_final=np.vstack((data_final, data_temp2))

  return data_final

def segment_speech(data_trial, segment_length):
  repeat=[]
  data_final=np.array([])
  for i in tqdm(range(len(data_trial))):
    repeat.append(int(data_trial[i].T.shape[1]/segment_length))   
    data_temp=data_trial[i].T
    data_temp2=np.array([])
    for j in tqdm(range(int(data_temp.shape[1]/segment_length))):
      llim=j*500
      data_temp1=data_temp[:,llim:llim+segment_length]
      if j==0:
        data_temp2=data_temp1[np.newaxis,:,:]
      else:
        data_temp2=np.vstack((data_temp2, data_temp1[np.newaxis,:,:]))
        print(data_temp2.shape)
    if i==0:
      data_final=data_temp2
      print("no")
      print(data_final.shape)
    else:
      print("yes")
      print(data_final.shape)
      data_final=np.vstack((data_final, data_temp2))
      #print(data_final.shape)
  return data_final, repeat

def repeater(label_vovel, rep):
  label_vovel_f=np.array([])
  for j in range(len(label_vovel)):
    temp=np.repeat(label_vovel[j], rep[j])
    if(j==0):
      label_vovel_f=temp
    else:
      label_vovel_f=np.append(label_vovel_f, temp)
  #TWST
  return label_vovel_f


