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
from preprocessed import *
from preprocess_individual import *

args=my_args()

def createFV_individual(data_train, data_test,f_split, fs, l_feat, c_ref):

  data_2_subs, data_2_subs_t, data_hilbert, data_hilbert_t = preprocess(data_train, data_test,f_split, fs, c_ref)

  # #Extracting all the features and concatenating them 
  final = np.array([])
  for j in range(0, data_2_subs.shape[0]):
      data_trial=data_2_subs[j,:,:].T
      data_trial_h=data_hilbert[j,:,:].T
      #data_trial.shape
      data_trial_s=[]
      data_trial_s_h=[]
      for h in range(f_split):
        data_trial_s.append(data_trial[h*int(data_2_subs.shape[2]/f_split):(h+1)*int(data_2_subs.shape[2]/f_split),:])
        data_trial_s_h.append(data_trial_h[h*int(data_2_subs.shape[2]/f_split):(h+1)*int(data_2_subs.shape[2]/f_split),:])
        
        #print(data_trial_s1.shape)
        '''data_trial_s2=data_trial[int(data_2_subs.shape[2]/f_split):2*int(data_2_subs.shape[2]/f_split),:]
        #print(data_trial_s2.shape)
        data_trial_s3=data_trial[2*int(data_2_subs.shape[2]/f_split):3*int(data_2_subs.shape[2]/f_split),:]'''
      #print(data_trial_s3.shape)

      #AR Coefficients

      #from statsmodels.datasets.sunspots import load
      #data = load()
      ARFV=np.array([])

      for i in range(0, data_train.shape[1]):
          for x in range(f_split):
            rho, sigma = sm.regression.linear_model.burg(data_trial_s[x][:,i], order=2)
            '''rho2, sigma2 = sm.regression.linear_model.burg(data_trial_s2[:,i], order=2)
            rho3, sigma3 = sm.regression.linear_model.burg(data_trial_s3[:,i], order=2)'''
            ARFV=np.append(ARFV, (rho))

      #print(ARFV) 

      #Haar wavelet

      HWDFV=np.array([])
      for i in range(0, data_train.shape[1]):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          for x in range(f_split):
            coeffs = wavedec(data_trial_s[x][:,i], 'haar',level=6)
            cA6,cD6,cD5,cD4,cD3, cD2, cD1=coeffs
            cD6_a=autocorr(cD6)
            cD5_a=autocorr(cD5)
            cD4_a=autocorr(cD4)
            cA=[np.var(cD1),np.var(cD2),np.var(cD3),np.var(cD4_a),np.var(cD5_a),np.var(cD6_a), np.mean(np.absolute(cD1)), np.mean(np.absolute(cD2)), np.mean(np.absolute(cD3))]
            HWDFV=np.append(HWDFV, cA)

      '''HWDFV1=np.array([])
      for i in range(0, data_train.shape[1]):
          (cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          HWDFV1=np.append(HWDFV1, cA)'''


      #Spectral Power estimates
      SPFV=np.array([])
      for i in range(0, data_train.shape[1]):
          for x in range(f_split):
            f, Pxx_den = signal.welch(data_trial_s[x][:,i], fs)
            SPFV=np.append(SPFV, (Pxx_den[np.where(f<100.0)[0].tolist()]))
          
      '''HUFV = np.array([])    
      for i in range(0, data_train.shape[1]):
        hu=pyeeg.hurst(data_trial[:,i])
        HUFV = np.append(HUFV, hu)'''
      
      PFDFV = np.array([])    
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          pfd=pyeeg.pfd(data_trial_s[x][:,i])
          PFDFV = np.append(PFDFV, pfd)
      #Concatenaton of All the feature vectors
      
      DFAFV = np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          dfa=pyeeg.dfa(data_trial_s[x][:,i])
          DFAFV = np.append(DFAFV, dfa)
        
      MNFV = np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          mn=np.mean(data_trial_s[x][:,i])
          MNFV = np.append(MNFV, mn)
        
      STDFV = np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          sd=np.std(data_trial_s[x][:,i])
          STDFV = np.append(STDFV, sd)
        
      MT1FV=np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean1(data_trial_s[x][:,i])
          MT1FV = np.append(MT1FV, f)
          
      MT2FV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean2(data_trial_s[x][:,i])
          MT2FV = np.append(MT2FV, f)
          
      LDFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = log_detec(data_trial_s[x][:,i])
          LDFV = np.append(LDFV, f)
          
      MDNFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = np.median(np.absolute(data_trial_s[x][:,i]))
          MDNFV = np.append(MDNFV, f)
          
      ABDFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = abs_diff(data_trial_s[x][:,i])
          ABDFV = np.append(ABDFV, f)
          
      MFQFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean_freq(data_trial_s[x][:,i])
          MFQFV = np.append(MFQFV, f)
          
      FAMFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = freq_atmax(data_trial_s[x][:,i])
          FAMFV = np.append(FAMFV, f)
          
      MPSFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = max_psd(data_trial_s[x][:,i])
          MPSFV = np.append(MPSFV, f)
          
      MT1FVH=np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean1(data_trial_s_h[x][:,i])
          MT1FVH = np.append(MT1FVH, f)
          
      MT2FVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean2(data_trial_s_h[x][:,i])
          MT2FVH = np.append(MT2FVH, f)
          
      LDFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = log_detec(data_trial_s_h[x][:,i])
          LDFVH = np.append(LDFVH, f)
          
      MDNFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = np.median(np.absolute(data_trial_s_h[x][:,i]))
          MDNFVH = np.append(MDNFVH, f)
          
      ABDFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = abs_diff(data_trial_s_h[x][:,i])
          ABDFVH = np.append(ABDFVH, f)
          
      MFQFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean_freq(data_trial_s_h[x][:,i])
          MFQFVH = np.append(MFQFVH, f)
          
      FAMFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = freq_atmax(data_trial_s_h[x][:,i])
          FAMFVH = np.append(FAMFVH, f)
          
      MPSFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = max_psd(data_trial_s_h[x][:,i])
          MPSFVH = np.append(MPSFVH, f)
          
          
          
      '''CORFV = np.array([])
      for i in range(0, data_train.shape[1]):
        cor=nolds.corr_dim(data_trial[:,i],1)
        CORFV = np.append(CORFV, sd)'''
        
      '''HJFV = np.array([])
      for i in range(0, data_train.shape[1]):
        hj=pyeeg.hjorth(data_trial[:,i],1)
        HJFV = np.append(HJFV, hj)'''
      
      concated=np.concatenate((ARFV,HWDFV,SPFV,PFDFV,DFAFV,MNFV,STDFV,MT1FV,MT2FV,LDFV, MDNFV, ABDFV, MFQFV, FAMFV, MPSFV,MT1FVH,MT2FVH,LDFVH,MDNFVH,ABDFVH,MFQFVH,FAMFVH,MPSFVH), axis=None)
      concated=np.reshape(concated, (-1, 1))
      if j==0:
          final=concated
      else:
          final= np.hstack((final, concated))
      print(j)
  print("THE NEW FEATURES")
  print(final.shape)

  final=final.T

  eegData=np.rollaxis(data_2_subs, 0, 3)
  eegData_ls=[]
  for d in range(f_split):
    eegData_ls.append(eegData[:,d*int(data_2_subs.shape[2]/f_split):(d+1)*int(data_2_subs.shape[2]/f_split),:])
                              
  #eegData.shape



  # Subband Information Quantity
  # delta (0.5–4 Hz)
  for x in range(f_split):                           
    #eegData_delta = eeg.filt_data(eegData, 0.5, 4, fs)
    eegData_delta = eeg.filt_data(eegData_ls[x], 0.5, 4, fs)
    eegData_theta = eeg.filt_data(eegData_ls[x], 4, 8, fs)
    eegData_alpha = eeg.filt_data(eegData_ls[x], 8, 12, fs)
    eegData_beta = eeg.filt_data(eegData_ls[x], 12, 30, fs)
    eegData_gamma = eeg.filt_data(eegData_ls[x], 30, 80, fs)
    #eegData_delta_l.append(
    if(x==0):
      ShannonRes_delta = eeg.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_theta = eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_alpha = eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_beta = eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_gamma = eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2).T
    else:
      ShannonRes_delta = np.hstack((ShannonRes_delta, eeg.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_theta = np.hstack((ShannonRes_theta, eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_alpha = np.hstack((ShannonRes_alpha, eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_beta = np.hstack((ShannonRes_beta, eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_gamma = np.hstack((ShannonRes_gamma, eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2).T))
    
  # theta (4–8 Hz)
  '''eegData_theta = eeg.filt_data(eegData, 4, 8, fs)
  ShannonRes_theta = eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2)
  # alpha (8–12 Hz)
  eegData_alpha = eeg.filt_data(eegData, 8, 12, fs)
  ShannonRes_alpha = eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2)
  # beta (12–30 Hz)
  eegData_beta = eeg.filt_data(eegData, 12, 30, fs)
  ShannonRes_beta = eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2)
  # gamma (30–100 Hz)
  eegData_gamma = eeg.filt_data(eegData, 30, 80, fs)
  ShannonRes_gamma = eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2)'''

  # Hjorth Mobility
  # Hjorth Complexity
  for x in range(f_split):
    #HjorthMob, HjorthComp = eeg.hjorthParameters(eegData)
    if(x==0):
      HjorthMob=eeg.hjorthParameters(eegData_ls[x])[0].T
      HjorthComp=eeg.hjorthParameters(eegData_ls[x])[1].T
    else:
      HjorthMob=np.hstack((HjorthMob,eeg.hjorthParameters(eegData_ls[x])[0].T))
      HjorthComp=np.hstack((HjorthComp,eeg.hjorthParameters(eegData_ls[x])[1].T))
  # Median Frequency
  for x in range(f_split):
    if(x==0):
      medianFreqRes = eeg.medianFreq(eegData_ls[x],fs).T
    else:
      medianFreqRes = np.hstack((medianFreqRes, eeg.medianFreq(eegData_ls[x],fs).T))
      
  for x in range(f_split):
    if(x==0):
      std_res = eeg.eegStd(eegData_ls[x]).T
    else:
      std_res = np.hstack((std_res, eeg.eegStd(eegData_ls[x]).T))
  # Standard Deviation
  
  for x in range(f_split):
    if(x==0):
      regularity_res = eeg.eegRegularity(eegData_ls[x],fs).T
    else:
      regularity_res = np.hstack((regularity_res, eeg.eegRegularity(eegData_ls[x],fs).T))
  # Regularity (burst-suppression)
  #regularity_res = eeg.eegRegularity(eegData,fs)

  # Spikes
  minNumSamples = int(70*fs/1000)
  for x in range(f_split):
    if(x==0):
      spikeNum_res = eeg.spikeNum(eegData_ls[x],minNumSamples).T
    else:
      spikeNum_res = np.hstack((spikeNum_res, eeg.spikeNum(eegData_ls[x],minNumSamples).T))
                              
  

  # Sharp spike
  for x in range(f_split):
    if(x==0):
      sharpSpike_res = eeg.shortSpikeNum(eegData_ls[x],minNumSamples).T
    else:
      sharpSpike_res = np.hstack((sharpSpike_res, eeg.spikeNum(eegData_ls[x],minNumSamples).T))
  
  
  #remove from here REMPOMOMOPMPOM
  concated=np.concatenate((ShannonRes_delta, ShannonRes_theta, ShannonRes_alpha, ShannonRes_beta, ShannonRes_gamma, HjorthMob, HjorthComp, medianFreqRes, std_res, regularity_res, spikeNum_res, sharpSpike_res), axis=1)

  final=np.hstack((final, concated))

  # δ band Power
  #bandPwr_delta = eeg.bandPower(eegData, 0.5, 4, fs)
  # θ band Power
  #bandPwr_theta = eeg.bandPower(eegData, 4, 8, fs)
  # α band Power
  #bandPwr_alpha = eeg.bandPower(eegData, 8, 12, fs)
  # β band Power
  #bandPwr_beta = eeg.bandPower(eegData, 12, 30, fs)
  # γ band Power
  for x in range(f_split):
    if(x==0):
      bandPwr_gamma = eeg.bandPower(eegData_ls[x], 30, 80, fs).T
    else:
      bandPwr_gamma = np.hstack((bandPwr_gamma, eeg.bandPower(eegData_ls[x], 30, 80, fs).T))
  
  

  concated_n=bandPwr_gamma
  final=np.hstack((final, concated_n))

  '''HTFV=np.array([])
  for j in range(0, eegData.shape[2]):
    eegData_temp=eegData[:,:,j]
    HTFV_temp=np.array([])
    for i in range(0, eegData.shape[0]):
      HTFV_temp=np.append(HTFV_temp, np.imag(hilbert(eegData_temp[i,:])))
    if(j==0):
      HTFV=HTFV_temp
    else:
      HTFV=np.vstack((HTFV, HTFV_temp))
    print(j)

  final=np.hstack((final, HTFV))
  #final.shape'''


  final_t = np.array([])
  for j in range(0 ,data_2_subs_t.shape[0]):
      data_trial=data_2_subs_t[j,:,:].T
      data_trial_h=data_hilbert_t[j,:,:].T
      #data_trial.shape
      data_trial_s=[]
      data_trial_s_h=[]
      for h in range(f_split):
        data_trial_s.append(data_trial[h*int(data_2_subs_t.shape[2]/f_split):(h+1)*int(data_2_subs_t.shape[2]/f_split),:])
        data_trial_s_h.append(data_trial_h[h*int(data_2_subs_t.shape[2]/f_split):(h+1)*int(data_2_subs_t.shape[2]/f_split),:])

      '''data_trial_s1=data_trial[0:int(data_2_subs_t.shape[2]/3),:]
      #print(data_trial_s1.shape)
      data_trial_s2=data_trial[int(data_2_subs_t.shape[2]/3):2*int(data_2_subs_t.shape[2]/3),:]
      #print(data_trial_s2.shape)
      data_trial_s3=data_trial[2*int(data_2_subs_t.shape[2]/3):3*int(data_2_subs_t.shape[2]/3),:]
      #print(data_trial_s3.shape)'''

      #AR Coefficients
      #from statsmodels.datasets.sunspots import load
      #data = load()
                              
      ARFV=np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
          for x in range(f_split):
            rho, sigma = sm.regression.linear_model.burg(data_trial_s[x][:,i], order=2)
            '''rho2, sigma2 = sm.regression.linear_model.burg(data_trial_s2[:,i], order=2)
            rho3, sigma3 = sm.regression.linear_model.burg(data_trial_s3[:,i], order=2)'''
            ARFV=np.append(ARFV, (rho))

      #print(ARFV) 

      #Haar wavelet

      HWDFV=np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          for x in range(f_split):
            coeffs = wavedec(data_trial_s[x][:,i], 'haar',level=6)
            cA6,cD6,cD5,cD4,cD3, cD2, cD1=coeffs
            cD6_a=autocorr(cD6)
            cD5_a=autocorr(cD5)
            cD4_a=autocorr(cD4)
            cA=[np.var(cD1),np.var(cD2),np.var(cD3),np.var(cD4_a),np.var(cD5_a),np.var(cD6_a), np.mean(np.absolute(cD1)), np.mean(np.absolute(cD2)), np.mean(np.absolute(cD3))]
            HWDFV=np.append(HWDFV, cA)

      '''HWDFV1=np.array([])
      for i in range(0, data_train.shape[1]):
          (cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          HWDFV1=np.append(HWDFV1, cA)'''


      #Spectral Power estimates
      SPFV=np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
          for x in range(f_split):
            f, Pxx_den = signal.welch(data_trial_s[x][:,i], fs)
            SPFV=np.append(SPFV, (Pxx_den[np.where(f<100.0)[0].tolist()]))
          
      '''HUFV = np.array([])    
      for i in range(0, data_train.shape[1]):
        hu=pyeeg.hurst(data_trial[:,i])
        HUFV = np.append(HUFV, hu)'''
      
      PFDFV = np.array([])    
      for i in range(0, data_2_subs_t.shape[1]):
        for x in range(f_split):
          pfd=pyeeg.pfd(data_trial_s[x][:,i])
          PFDFV = np.append(PFDFV, pfd)
      #Concatenaton of All the feature vectors
      
      DFAFV = np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
        for x in range(f_split):
          dfa=pyeeg.dfa(data_trial_s[x][:,i])
          DFAFV = np.append(DFAFV, dfa)
        
        
      MNFV = np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
        for x in range(f_split):
          mn=np.mean(data_trial_s[x][:,i])
          MNFV = np.append(MNFV, mn)
        
      STDFV = np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
        for x in range(f_split):
          sd=np.std(data_trial_s[x][:,i])
          STDFV = np.append(STDFV, sd)
        
      MT1FV=np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean1(data_trial_s[x][:,i])
          MT1FV = np.append(MT1FV, f)
          
      MT2FV=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean2(data_trial_s[x][:,i])
          MT2FV = np.append(MT2FV, f)
          
      LDFV=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = log_detec(data_trial_s[x][:,i])
          LDFV = np.append(LDFV, f)
          
      MDNFV=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = np.median(np.absolute(data_trial_s[x][:,i]))
          MDNFV = np.append(MDNFV, f)
          
      ABDFV=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = abs_diff(data_trial_s[x][:,i])
          ABDFV = np.append(ABDFV, f)
          
      MFQFV=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean_freq(data_trial_s[x][:,i])
          MFQFV = np.append(MFQFV, f)
          
      FAMFV=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = freq_atmax(data_trial_s[x][:,i])
          FAMFV = np.append(FAMFV, f)
          
      MPSFV=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = max_psd(data_trial_s[x][:,i])
          MPSFV = np.append(MPSFV, f)
                              
      MT1FVH=np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean1(data_trial_s_h[x][:,i])
          MT1FVH = np.append(MT1FVH, f)
          
      MT2FVH=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean2(data_trial_s_h[x][:,i])
          MT2FVH = np.append(MT2FVH, f)
          
      LDFVH=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = log_detec(data_trial_s_h[x][:,i])
          LDFVH = np.append(LDFVH, f)
          
      MDNFVH=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = np.median(np.absolute(data_trial_s_h[x][:,i]))
          MDNFVH = np.append(MDNFVH, f)
          
      ABDFVH=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = abs_diff(data_trial_s_h[x][:,i])
          ABDFVH = np.append(ABDFVH, f)
          
      MFQFVH=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean_freq(data_trial_s_h[x][:,i])
          MFQFVH = np.append(MFQFVH, f)
          
      FAMFVH=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = freq_atmax(data_trial_s_h[x][:,i])
          FAMFVH = np.append(FAMFVH, f)
          
      MPSFVH=np.array([])
      for i in tqdm(range(0, data_2_subs_t.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = max_psd(data_trial_s_h[x][:,i])
          MPSFVH = np.append(MPSFVH, f)
                              
         
        
      
      '''HJFV = np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
        hj=pyeeg.hjorth(data_trial[:,i],1)
        HJFV = np.append(HJFV, hj)'''
      
      
      
      #Spectral Power estimates
      

      #Concatenaton of All the feature vectors
      concated=np.concatenate((ARFV,HWDFV,SPFV,PFDFV,DFAFV,MNFV,STDFV,MT1FV,MT2FV,LDFV, MDNFV, ABDFV, MFQFV, FAMFV, MPSFV,MT1FVH,MT2FVH,LDFVH,MDNFVH,ABDFVH,MFQFVH,FAMFVH,MPSFVH), axis=None)
      concated=np.reshape(concated, (-1, 1))
      if j==0:
          final_t=concated
      else:
          final_t= np.hstack((final_t, concated))
      print(j)
  print(final_t.shape)

  final_t=final_t.T
  final_t.shape

  eegData_t=np.rollaxis(data_2_subs_t, 0, 3)
  eegData_ls_t=[]
  for d in range(f_split):
    eegData_ls_t.append(eegData_t[:,d*int(data_2_subs_t.shape[2]/f_split):(d+1)*int(data_2_subs_t.shape[2]/f_split),:])
  #eegData_t.shape
                              
  for x in range(f_split):                           
    #eegData_delta = eeg.filt_data(eegData, 0.5, 4, fs)
    eegData_delta_t = eeg.filt_data(eegData_ls_t[x], 0.5, 4, fs)
    eegData_theta_t = eeg.filt_data(eegData_ls_t[x], 4, 8, fs)
    eegData_alpha_t = eeg.filt_data(eegData_ls_t[x], 8, 12, fs)
    eegData_beta_t = eeg.filt_data(eegData_ls_t[x], 12, 30, fs)
    eegData_gamma_t = eeg.filt_data(eegData_ls_t[x], 30, 80, fs)
      #eegData_delta_l.append(
    if(x==0):
      ShannonRes_delta_t = eeg.shannonEntropy(eegData_delta_t, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_theta_t = eeg.shannonEntropy(eegData_theta_t, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_alpha_t = eeg.shannonEntropy(eegData_alpha_t, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_beta_t = eeg.shannonEntropy(eegData_beta_t, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_gamma_t = eeg.shannonEntropy(eegData_gamma_t, bin_min=-200, bin_max=200, binWidth=2).T
    else:
      ShannonRes_delta_t = np.hstack((ShannonRes_delta_t, eeg.shannonEntropy(eegData_delta_t, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_theta_t = np.hstack((ShannonRes_theta_t, eeg.shannonEntropy(eegData_theta_t, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_alpha_t = np.hstack((ShannonRes_alpha_t, eeg.shannonEntropy(eegData_alpha_t, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_beta_t = np.hstack((ShannonRes_beta_t, eeg.shannonEntropy(eegData_beta_t, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_gamma_t = np.hstack((ShannonRes_gamma_t, eeg.shannonEntropy(eegData_gamma_t, bin_min=-200, bin_max=200, binWidth=2).T))


  # Subband Information Quantity
  # delta (0.5–4 Hz)
  '''eegData_delta_t = eeg.filt_data(eegData_t, 0.5, 4, fs)
  ShannonRes_delta_t = eeg.shannonEntropy(eegData_delta_t, bin_min=-200, bin_max=200, binWidth=2)
  # theta (4–8 Hz)
  eegData_theta_t = eeg.filt_data(eegData_t, 4, 8, fs)
  ShannonRes_theta_t = eeg.shannonEntropy(eegData_theta_t, bin_min=-200, bin_max=200, binWidth=2)
  # alpha (8–12 Hz)
  eegData_alpha_t = eeg.filt_data(eegData_t, 8, 12, fs)
  ShannonRes_alpha_t = eeg.shannonEntropy(eegData_alpha_t, bin_min=-200, bin_max=200, binWidth=2)
  # beta (12–30 Hz)
  eegData_beta_t = eeg.filt_data(eegData_t, 12, 30, fs)
  ShannonRes_beta_t = eeg.shannonEntropy(eegData_beta_t, bin_min=-200, bin_max=200, binWidth=2)
  # gamma (30–100 Hz)
  eegData_gamma_t = eeg.filt_data(eegData_t, 30, 80, fs)
  ShannonRes_gamma_t = eeg.shannonEntropy(eegData_gamma_t, bin_min=-200, bin_max=200, binWidth=2)'''
                              
  # Hjorth Mobility
  # Hjorth Complexity
  for x in range(f_split):
    #HjorthMob, HjorthComp = eeg.hjorthParameters(eegData)
    if(x==0):
      HjorthMob_t=eeg.hjorthParameters(eegData_ls_t[x])[0].T
      HjorthComp_t=eeg.hjorthParameters(eegData_ls_t[x])[1].T
    else:
      HjorthMob_t=np.hstack((HjorthMob_t,eeg.hjorthParameters(eegData_ls_t[x])[0].T))
      HjorthComp_t=np.hstack((HjorthComp_t,eeg.hjorthParameters(eegData_ls_t[x])[1].T))
  # Median Frequency
  for x in range(f_split):
    if(x==0):
      medianFreqRes_t = eeg.medianFreq(eegData_ls_t[x],fs).T
    else:
      medianFreqRes_t = np.hstack((medianFreqRes_t, eeg.medianFreq(eegData_ls_t[x],fs).T))
      
  for x in range(f_split):
    if(x==0):
      std_res_t = eeg.eegStd(eegData_ls_t[x]).T
    else:
      std_res_t = np.hstack((std_res_t, eeg.eegStd(eegData_ls_t[x]).T))
  # Standard Deviation
  
  for x in range(f_split):
    if(x==0):
      regularity_res_t = eeg.eegRegularity(eegData_ls_t[x],fs).T
    else:
      regularity_res_t = np.hstack((regularity_res_t, eeg.eegRegularity(eegData_ls_t[x],fs).T))
  # Regularity (burst-suppression)
  #regularity_res = eeg.eegRegularity(eegData,fs)

  # Spikes
  minNumSamples = int(70*fs/1000)
  for x in range(f_split):
    if(x==0):
      spikeNum_res_t = eeg.spikeNum(eegData_ls_t[x],minNumSamples).T
    else:
      spikeNum_res_t = np.hstack((spikeNum_res_t, eeg.spikeNum(eegData_ls_t[x],minNumSamples).T))
                              
  

  # Sharp spike
  for x in range(f_split):
    if(x==0):
      sharpSpike_res_t = eeg.shortSpikeNum(eegData_ls_t[x],minNumSamples).T
    else:
      sharpSpike_res_t = np.hstack((sharpSpike_res_t, eeg.spikeNum(eegData_ls_t[x],minNumSamples).T))

  # Hjorth Mobility
  # Hjorth Complexity
  '''HjorthMob_t, HjorthComp_t = eeg.hjorthParameters(eegData_t)


  # Median Frequency
  medianFreqRes_t = eeg.medianFreq(eegData_t,fs)

  # Standard Deviation
  std_res_t = eeg.eegStd(eegData_t)

  # Regularity (burst-suppression)
  regularity_res_t = eeg.eegRegularity(eegData_t,fs)

  # Spikes
  minNumSamples = int(70*fs/1000)
  spikeNum_res_t = eeg.spikeNum(eegData_t,minNumSamples)


  # Sharp spike
  sharpSpike_res_t = eeg.shortSpikeNum(eegData_t,minNumSamples)'''
  #REMPMVPOMMPOMPOMPOMPOMOMPOMPOM
  concated_t=np.concatenate(( ShannonRes_delta_t, ShannonRes_theta_t, ShannonRes_alpha_t, ShannonRes_beta_t, ShannonRes_gamma_t, HjorthMob_t, HjorthComp_t, medianFreqRes_t, std_res_t, regularity_res_t, spikeNum_res_t, sharpSpike_res_t), axis=1)

  final_t=np.hstack((final_t, concated_t))

  # δ band Power
  '''bandPwr_delta_t = eeg.bandPower(eegData_t, 0.5, 4, fs)
  #too large
  # θ band Power
  bandPwr_theta_t = eeg.bandPower(eegData_t, 4, 8, fs)
  #too large
  # α band Power
  bandPwr_alpha_t = eeg.bandPower(eegData_t, 8, 12, fs)
  #too large
  # β band Power
  bandPwr_beta_t = eeg.bandPower(eegData_t, 12, 30, fs)'''
  #too large
  # γ band Power
  for x in range(f_split):
    if(x==0):
      bandPwr_gamma_t = eeg.bandPower(eegData_ls_t[x], 30, 80, fs).T
    else:
      bandPwr_gamma_t = np.hstack((bandPwr_gamma_t, eeg.bandPower(eegData_ls_t[x], 30, 80, fs).T))
  #bandPwr_gamma_t = eeg.bandPower(eegData_t, 30, 80, fs)

  concated_n_t= bandPwr_gamma_t
  final_t=np.hstack((final_t, concated_n_t))

  '''HTFV_t=np.array([])
  for j in range(0, eegData_t.shape[2]):
    eegData_temp=eegData_t[:,:,j]
    HTFV_temp=np.array([])
    for i in range(0, eegData.shape[0]):
      HTFV_temp=np.append(HTFV_temp, np.imag(hilbert(eegData_temp[i,:])))
    if(j==0):
      HTFV_t=HTFV_temp
    else:
      HTFV_t=np.vstack((HTFV_t, HTFV_temp))
  #final_t.shape
  final_t=np.hstack((final_t, HTFV_t))'''

  #MT1FV,MT2FV,LDFV, MDNFV, ABDFV, MFQFV, FAMFV, MPSFV,MT1FVH,MT2FVH,LDFVH,MDNFVH,ABDFVH,MFQFVH,FAMFVH,MPSFVH
  #importance per feature
  nfeatures_1=ARFV.shape[0]
  nfeatures_2=HWDFV.shape[0]
  #nfeatures_3=HWDFV1.shape[0]
  nfeatures_3=SPFV.shape[0]
  #nfeatures_4=HUFV.shape[0]
  nfeatures_4=PFDFV.shape[0]
  nfeatures_5=DFAFV.shape[0]
  nfeatures_6=MNFV.shape[0]
  nfeatures_7=STDFV.shape[0]
  #nfeatures_9=CORFV.shape[0]
  nfeatures_8=MT1FV.shape[0]
  nfeatures_9=MT2FV.shape[0]
  nfeatures_10=LDFV.shape[0]
  nfeatures_11=MDNFV.shape[0]
  nfeatures_12=ABDFV.shape[0]
  nfeatures_13=MFQFV.shape[0]
  nfeatures_14=FAMFV.shape[0]
  nfeatures_15=MPSFV.shape[0]
  nfeatures_16=MT1FVH.shape[0]
  nfeatures_17=MT2FVH.shape[0]
  nfeatures_18=LDFVH.shape[0]
  nfeatures_19=MDNFVH.shape[0]
  nfeatures_20=ABDFVH.shape[0]
  nfeatures_21=MFQFVH.shape[0]
  nfeatures_22=FAMFVH.shape[0]
  nfeatures_23=MPSFVH.shape[0]
  

  #EEG EXTRACT FEATURES
  nfeatures_24=ShannonRes_delta.shape[1]
  nfeatures_25=ShannonRes_theta.shape[1]
  nfeatures_26=ShannonRes_alpha.shape[1]
  nfeatures_27=ShannonRes_beta.shape[1]
  nfeatures_28=ShannonRes_gamma.shape[1]
  nfeatures_29=HjorthMob.shape[1]
  nfeatures_30=HjorthComp.shape[1]
  nfeatures_31=medianFreqRes.shape[1]
  nfeatures_32=std_res.shape[1]
  nfeatures_33=regularity_res.shape[1]
  nfeatures_34=spikeNum_res.shape[1]
  nfeatures_35=sharpSpike_res.shape[1]
  nfeatures_36=bandPwr_gamma.shape[1]
  #nfeatures_38=HTFV_temp.shape[0]
  '''nfeatures_16=bandPwr_alpha.shape[0]
  nfeatures_17=bandPwr_beta.shape[0]
  nfeatures_18=bandPwr_gamma.shape[0]
  nfeatures_19=HTFV_temp.shape[0]'''

  llim1=0
  llim2=llim1+nfeatures_1
  llim3=llim2+nfeatures_2
  llim4=llim3+nfeatures_3
  llim5=llim4+nfeatures_4
  llim6=llim5+nfeatures_5
  llim7=llim6+nfeatures_6
  llim8=llim7+nfeatures_7
  llim9=llim8+nfeatures_8
  llim10=llim9+nfeatures_9
  llim11=llim10+nfeatures_10
  llim12=llim11+nfeatures_11
  llim13=llim12+nfeatures_12
  llim14=llim13+nfeatures_13
  llim15=llim14+nfeatures_14
  llim16=llim15+nfeatures_15
  llim17=llim16+nfeatures_16
  llim18=llim17+nfeatures_17
  llim17=llim16+nfeatures_16
  llim18=llim17+nfeatures_17
  llim19=llim18+nfeatures_18
  llim20=llim19+nfeatures_19
  llim21=llim20+nfeatures_20
  llim22=llim21+nfeatures_21
  llim23=llim22+nfeatures_22
  llim24=llim23+nfeatures_23
  llim24=llim23+nfeatures_23
  llim25=llim24+nfeatures_24
  llim26=llim25+nfeatures_25
  llim27=llim26+nfeatures_26
  llim28=llim27+nfeatures_27
  llim29=llim28+nfeatures_28
  llim30=llim29+nfeatures_29
  llim31=llim30+nfeatures_30
  llim32=llim31+nfeatures_31
  llim33=llim32+nfeatures_32
  llim34=llim33+nfeatures_33
  llim35=llim34+nfeatures_34
  llim36=llim35+nfeatures_35
  llim37=llim36+nfeatures_36
  #llim38=llim37+nfeatures_37
  #llm39=llim38+nfeatures_38
  #llim23=llim22+nfeatures_22
  #llim24=llim23+nfeatures_23
  #llim25=llim24+nfeatures_24

  llim=[llim1, llim2, llim3, llim4, llim5, llim6, llim7, llim8, llim9, llim10, llim11, llim12, llim13, llim14, llim15, llim16, llim17, llim18,llim19,llim20,llim21,llim22,llim23,llim24,llim25,llim26,llim27,llim28,llim29,llim30,llim31,llim32,llim33,llim34,llim35,llim36,llim37]
  nfeatures=[nfeatures_1, nfeatures_2,nfeatures_3,nfeatures_4,nfeatures_5,nfeatures_6,nfeatures_7,nfeatures_8,nfeatures_9,nfeatures_10,nfeatures_11,nfeatures_12,nfeatures_13,nfeatures_14,nfeatures_15,nfeatures_16,nfeatures_17,nfeatures_18,nfeatures_19,nfeatures_20,nfeatures_21,nfeatures_22,nfeatures_23,nfeatures_24,nfeatures_25,nfeatures_26,nfeatures_27,nfeatures_28,nfeatures_29,nfeatures_30,nfeatures_31,nfeatures_32,nfeatures_33,nfeatures_34,nfeatures_35,nfeatures_36]

  for i, lf in enumerate(l_feat):
    print("trial"+str(lf))
    if(i==0):
      nump=np.arange(llim[lf], llim[lf]+nfeatures[lf])
      #=[]
    else:
      numpu=np.arange(llim[lf], llim[lf]+nfeatures[lf])
      nump=np.append(nump, numpu)
  numpl=nump.tolist()
  numpl = list(map(int,numpl))

  print(final.shape)

  final=final[:,numpl]
  final_t=final_t[:,numpl]

  print(final.shape)

  list_rand=[]
  for i in range(0,final.shape[1]):
    list_rand.append("c"+str(i))
  len(list_rand)
  df_train = pd.DataFrame(final, columns = list_rand)
  df_test = pd.DataFrame(final_t, columns = list_rand)

  return df_train, df_test

def createFV_individual_feat(data_train, f_split,fs, l_feat, c_ref):
#subsampling by 4 
  data_2_subs,  data_hilbert = preprocess_individual(data_train, f_split, fs, c_ref) 
  #Extracting all the features and concatenating them 
  final = np.array([])
  for j in range(0, data_2_subs.shape[0]):
      data_trial=data_2_subs[j,:,:].T
      data_trial_h=data_hilbert[j,:,:].T
      #data_trial.shape
      data_trial_s=[]
      data_trial_s_h=[]
      for h in range(f_split):
        data_trial_s.append(data_trial[h*int(data_2_subs.shape[2]/f_split):(h+1)*int(data_2_subs.shape[2]/f_split),:])
        data_trial_s_h.append(data_trial_h[h*int(data_2_subs.shape[2]/f_split):(h+1)*int(data_2_subs.shape[2]/f_split),:])
        
        #print(data_trial_s1.shape)
        '''data_trial_s2=data_trial[int(data_2_subs.shape[2]/f_split):2*int(data_2_subs.shape[2]/f_split),:]
        #print(data_trial_s2.shape)
        data_trial_s3=data_trial[2*int(data_2_subs.shape[2]/f_split):3*int(data_2_subs.shape[2]/f_split),:]'''
      #print(data_trial_s3.shape)

      #AR Coefficients

      #from statsmodels.datasets.sunspots import load
      #data = load()
      ARFV=np.array([])

      for i in range(0, data_train.shape[1]):
          for x in range(f_split):
            rho, sigma = sm.regression.linear_model.burg(data_trial_s[x][:,i], order=2)
            '''rho2, sigma2 = sm.regression.linear_model.burg(data_trial_s2[:,i], order=2)
            rho3, sigma3 = sm.regression.linear_model.burg(data_trial_s3[:,i], order=2)'''
            ARFV=np.append(ARFV, (rho))

      #print(ARFV) 

      #Haar wavelet

      HWDFV=np.array([])
      for i in range(0, data_train.shape[1]):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          for x in range(f_split):
            coeffs = wavedec(data_trial_s[x][:,i], 'haar',level=6)
            cA6,cD6,cD5,cD4,cD3, cD2, cD1=coeffs
            cD6_a=autocorr(cD6)
            cD5_a=autocorr(cD5)
            cD4_a=autocorr(cD4)
            cA=[np.var(cD1),np.var(cD2),np.var(cD3),np.var(cD4_a),np.var(cD5_a),np.var(cD6_a), np.mean(np.absolute(cD1)), np.mean(np.absolute(cD2)), np.mean(np.absolute(cD3))]
            HWDFV=np.append(HWDFV, cA)

      '''HWDFV1=np.array([])
      for i in range(0, data_train.shape[1]):
          (cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          HWDFV1=np.append(HWDFV1, cA)'''


      #Spectral Power estimates
      SPFV=np.array([])
      for i in range(0, data_train.shape[1]):
          for x in range(f_split):
            f, Pxx_den = signal.welch(data_trial_s[x][:,i], fs)
            SPFV=np.append(SPFV, (Pxx_den[np.where(f<100.0)[0].tolist()]))
          
      '''HUFV = np.array([])    
      for i in range(0, data_train.shape[1]):
        hu=pyeeg.hurst(data_trial[:,i])
        HUFV = np.append(HUFV, hu)'''
      
      PFDFV = np.array([])    
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          pfd=pyeeg.pfd(data_trial_s[x][:,i])
          PFDFV = np.append(PFDFV, pfd)
      #Concatenaton of All the feature vectors
      
      DFAFV = np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          dfa=pyeeg.dfa(data_trial_s[x][:,i])
          DFAFV = np.append(DFAFV, dfa)
        
      MNFV = np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          mn=np.mean(data_trial_s[x][:,i])
          MNFV = np.append(MNFV, mn)
        
      STDFV = np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          sd=np.std(data_trial_s[x][:,i])
          STDFV = np.append(STDFV, sd)
        
      MT1FV=np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean1(data_trial_s[x][:,i])
          MT1FV = np.append(MT1FV, f)
          
      MT2FV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean2(data_trial_s[x][:,i])
          MT2FV = np.append(MT2FV, f)
          
      LDFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = log_detec(data_trial_s[x][:,i])
          LDFV = np.append(LDFV, f)
          
      MDNFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = np.median(np.absolute(data_trial_s[x][:,i]))
          MDNFV = np.append(MDNFV, f)
          
      ABDFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = abs_diff(data_trial_s[x][:,i])
          ABDFV = np.append(ABDFV, f)
          
      MFQFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean_freq(data_trial_s[x][:,i])
          MFQFV = np.append(MFQFV, f)
          
      FAMFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = freq_atmax(data_trial_s[x][:,i])
          FAMFV = np.append(FAMFV, f)
          
      MPSFV=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = max_psd(data_trial_s[x][:,i])
          MPSFV = np.append(MPSFV, f)
          
      MT1FVH=np.array([])
      for i in range(0, data_train.shape[1]):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean1(data_trial_s_h[x][:,i])
          MT1FVH = np.append(MT1FVH, f)
          
      MT2FVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean2(data_trial_s_h[x][:,i])
          MT2FVH = np.append(MT2FVH, f)
          
      LDFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = log_detec(data_trial_s_h[x][:,i])
          LDFVH = np.append(LDFVH, f)
          
      MDNFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = np.median(np.absolute(data_trial_s_h[x][:,i]))
          MDNFVH = np.append(MDNFVH, f)
          
      ABDFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = abs_diff(data_trial_s_h[x][:,i])
          ABDFVH = np.append(ABDFVH, f)
          
      MFQFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = mean_freq(data_trial_s_h[x][:,i])
          MFQFVH = np.append(MFQFVH, f)
          
      FAMFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = freq_atmax(data_trial_s_h[x][:,i])
          FAMFVH = np.append(FAMFVH, f)
          
      MPSFVH=np.array([])
      for i in tqdm(range(0, data_train.shape[1])):
        for x in range(f_split):
          #(cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          f = max_psd(data_trial_s_h[x][:,i])
          MPSFVH = np.append(MPSFVH, f)
          
          
          
      '''CORFV = np.array([])
      for i in range(0, data_train.shape[1]):
        cor=nolds.corr_dim(data_trial[:,i],1)
        CORFV = np.append(CORFV, sd)'''
        
      '''HJFV = np.array([])
      for i in range(0, data_train.shape[1]):
        hj=pyeeg.hjorth(data_trial[:,i],1)
        HJFV = np.append(HJFV, hj)'''
      
      concated=np.concatenate((ARFV,HWDFV,SPFV,PFDFV,DFAFV,MNFV,STDFV,MT1FV,MT2FV,LDFV, MDNFV, ABDFV, MFQFV, FAMFV, MPSFV,MT1FVH,MT2FVH,LDFVH,MDNFVH,ABDFVH,MFQFVH,FAMFVH,MPSFVH), axis=None)
      concated=np.reshape(concated, (-1, 1))
      if j==0:
          final=concated
      else:
          final= np.hstack((final, concated))
      print(j)
  print("THE NEW FEATURES")
  print(final.shape)

  final=final.T

  eegData=np.rollaxis(data_2_subs, 0, 3)
  eegData_ls=[]
  for d in range(f_split):
    eegData_ls.append(eegData[:,d*int(data_2_subs.shape[2]/f_split):(d+1)*int(data_2_subs.shape[2]/f_split),:])
                              
  #eegData.shape



  # Subband Information Quantity
  # delta (0.5–4 Hz)
  for x in range(f_split):                           
    #eegData_delta = eeg.filt_data(eegData, 0.5, 4, fs)
    eegData_delta = eeg.filt_data(eegData_ls[x], 0.5, 4, fs)
    eegData_theta = eeg.filt_data(eegData_ls[x], 4, 8, fs)
    eegData_alpha = eeg.filt_data(eegData_ls[x], 8, 12, fs)
    eegData_beta = eeg.filt_data(eegData_ls[x], 12, 30, fs)
    eegData_gamma = eeg.filt_data(eegData_ls[x], 30, 80, fs)
    #eegData_delta_l.append(
    if(x==0):
      ShannonRes_delta = eeg.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_theta = eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_alpha = eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_beta = eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2).T
      ShannonRes_gamma = eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2).T
    else:
      ShannonRes_delta = np.hstack((ShannonRes_delta, eeg.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_theta = np.hstack((ShannonRes_theta, eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_alpha = np.hstack((ShannonRes_alpha, eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_beta = np.hstack((ShannonRes_beta, eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2).T))
      ShannonRes_gamma = np.hstack((ShannonRes_gamma, eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2).T))
    
  # theta (4–8 Hz)
  '''eegData_theta = eeg.filt_data(eegData, 4, 8, fs)
  ShannonRes_theta = eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2)
  # alpha (8–12 Hz)
  eegData_alpha = eeg.filt_data(eegData, 8, 12, fs)
  ShannonRes_alpha = eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2)
  # beta (12–30 Hz)
  eegData_beta = eeg.filt_data(eegData, 12, 30, fs)
  ShannonRes_beta = eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2)
  # gamma (30–100 Hz)
  eegData_gamma = eeg.filt_data(eegData, 30, 80, fs)
  ShannonRes_gamma = eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2)'''

  # Hjorth Mobility
  # Hjorth Complexity
  for x in range(f_split):
    #HjorthMob, HjorthComp = eeg.hjorthParameters(eegData)
    if(x==0):
      HjorthMob=eeg.hjorthParameters(eegData_ls[x])[0].T
      HjorthComp=eeg.hjorthParameters(eegData_ls[x])[1].T
    else:
      HjorthMob=np.hstack((HjorthMob,eeg.hjorthParameters(eegData_ls[x])[0].T))
      HjorthComp=np.hstack((HjorthComp,eeg.hjorthParameters(eegData_ls[x])[1].T))
  # Median Frequency
  for x in range(f_split):
    if(x==0):
      medianFreqRes = eeg.medianFreq(eegData_ls[x],fs).T
    else:
      medianFreqRes = np.hstack((medianFreqRes, eeg.medianFreq(eegData_ls[x],fs).T))
      
  for x in range(f_split):
    if(x==0):
      std_res = eeg.eegStd(eegData_ls[x]).T
    else:
      std_res = np.hstack((std_res, eeg.eegStd(eegData_ls[x]).T))
  # Standard Deviation
  
  for x in range(f_split):
    if(x==0):
      regularity_res = eeg.eegRegularity(eegData_ls[x],fs).T
    else:
      regularity_res = np.hstack((regularity_res, eeg.eegRegularity(eegData_ls[x],fs).T))
  # Regularity (burst-suppression)
  #regularity_res = eeg.eegRegularity(eegData,fs)

  # Spikes
  minNumSamples = int(70*fs/1000)
  for x in range(f_split):
    if(x==0):
      spikeNum_res = eeg.spikeNum(eegData_ls[x],minNumSamples).T
    else:
      spikeNum_res = np.hstack((spikeNum_res, eeg.spikeNum(eegData_ls[x],minNumSamples).T))
                              
  

  # Sharp spike
  for x in range(f_split):
    if(x==0):
      sharpSpike_res = eeg.shortSpikeNum(eegData_ls[x],minNumSamples).T
    else:
      sharpSpike_res = np.hstack((sharpSpike_res, eeg.spikeNum(eegData_ls[x],minNumSamples).T))
  
  
  #remove from here REMPOMOMOPMPOM
  concated=np.concatenate((ShannonRes_delta, ShannonRes_theta, ShannonRes_alpha, ShannonRes_beta, ShannonRes_gamma, HjorthMob, HjorthComp, medianFreqRes, std_res, regularity_res, spikeNum_res, sharpSpike_res), axis=1)

  final=np.hstack((final, concated))

  # δ band Power
  #bandPwr_delta = eeg.bandPower(eegData, 0.5, 4, fs)
  # θ band Power
  #bandPwr_theta = eeg.bandPower(eegData, 4, 8, fs)
  # α band Power
  #bandPwr_alpha = eeg.bandPower(eegData, 8, 12, fs)
  # β band Power
  #bandPwr_beta = eeg.bandPower(eegData, 12, 30, fs)
  # γ band Power
  for x in range(f_split):
    if(x==0):
      bandPwr_gamma = eeg.bandPower(eegData_ls[x], 30, 80, fs).T
    else:
      bandPwr_gamma = np.hstack((bandPwr_gamma, eeg.bandPower(eegData_ls[x], 30, 80, fs).T))
  
  

  concated_n=bandPwr_gamma
  final=np.hstack((final, concated_n))

  '''HTFV=np.array([])
  for j in range(0, eegData.shape[2]):
    eegData_temp=eegData[:,:,j]
    HTFV_temp=np.array([])
    for i in range(0, eegData.shape[0]):
      HTFV_temp=np.append(HTFV_temp, np.imag(hilbert(eegData_temp[i,:])))
    if(j==0):
      HTFV=HTFV_temp
    else:
      HTFV=np.vstack((HTFV, HTFV_temp))
    print(j)

  final=np.hstack((final, HTFV))
  #final.shape'''




 

  #importance per feature
  nfeatures_1=ARFV.shape[0]
  nfeatures_2=HWDFV.shape[0]
  nfeatures_3=SPFV.shape[0]
  #nfeatures_4=HUFV.shape[0]
  nfeatures_4=PFDFV.shape[0]
  nfeatures_5=DFAFV.shape[0]
  nfeatures_6=MNFV.shape[0]
  nfeatures_7=STDFV.shape[0]
  #nfeatures_9=CORFV.shape[0]
  nfeatures_8=MT1FV.shape[0]
  nfeatures_9=MT2FV.shape[0]
  nfeatures_10=LDFV.shape[0]
  nfeatures_11=MDNFV.shape[0]
  nfeatures_12=ABDFV.shape[0]
  nfeatures_13=MFQFV.shape[0]
  nfeatures_14=FAMFV.shape[0]
  nfeatures_15=MPSFV.shape[0]
  nfeatures_16=MT1FVH.shape[0]
  nfeatures_17=MT2FVH.shape[0]
  nfeatures_18=LDFVH.shape[0]
  nfeatures_19=MDNFVH.shape[0]
  nfeatures_20=ABDFVH.shape[0]
  nfeatures_21=MFQFVH.shape[0]
  nfeatures_22=FAMFVH.shape[0]
  nfeatures_23=MPSFVH.shape[0]
  

  #EEG EXTRACT FEATURES
  nfeatures_24=ShannonRes_delta.shape[1]
  nfeatures_25=ShannonRes_theta.shape[1]
  nfeatures_26=ShannonRes_alpha.shape[1]
  nfeatures_27=ShannonRes_beta.shape[1]
  nfeatures_28=ShannonRes_gamma.shape[1]
  nfeatures_29=HjorthMob.shape[1]
  nfeatures_30=HjorthComp.shape[1]
  nfeatures_31=medianFreqRes.shape[1]
  nfeatures_32=std_res.shape[1]
  nfeatures_33=regularity_res.shape[1]
  nfeatures_34=spikeNum_res.shape[1]
  nfeatures_35=sharpSpike_res.shape[1]
  nfeatures_36=bandPwr_gamma.shape[1]
  #nfeatures_37=HTFV_temp.shape[0]
  '''nfeatures_16=bandPwr_alpha.shape[0]
  nfeatures_17=bandPwr_beta.shape[0]
  nfeatures_18=bandPwr_gamma.shape[0]
  nfeatures_19=HTFV_temp.shape[0]'''

  llim1=0
  llim2=llim1+nfeatures_1
  llim3=llim2+nfeatures_2
  llim4=llim3+nfeatures_3
  llim5=llim4+nfeatures_4
  llim6=llim5+nfeatures_5
  llim7=llim6+nfeatures_6
  llim8=llim7+nfeatures_7
  llim9=llim8+nfeatures_8
  llim10=llim9+nfeatures_9
  llim11=llim10+nfeatures_10
  llim12=llim11+nfeatures_11
  llim13=llim12+nfeatures_12
  llim14=llim13+nfeatures_13
  llim15=llim14+nfeatures_14
  llim16=llim15+nfeatures_15
  llim17=llim16+nfeatures_16
  llim18=llim17+nfeatures_17
  llim17=llim16+nfeatures_16
  llim18=llim17+nfeatures_17
  llim19=llim18+nfeatures_18
  llim20=llim19+nfeatures_19
  llim21=llim20+nfeatures_20
  llim22=llim21+nfeatures_21
  llim23=llim22+nfeatures_22
  llim24=llim23+nfeatures_23
  llim24=llim23+nfeatures_23
  llim25=llim24+nfeatures_24
  llim26=llim25+nfeatures_25
  llim27=llim26+nfeatures_26
  llim28=llim27+nfeatures_27
  llim29=llim28+nfeatures_28
  llim30=llim29+nfeatures_29
  llim31=llim30+nfeatures_30
  llim32=llim31+nfeatures_31
  llim33=llim32+nfeatures_32
  llim34=llim33+nfeatures_33
  llim35=llim34+nfeatures_34
  llim36=llim35+nfeatures_35
  llim37=llim36+nfeatures_36
  #llim38=llim37+nfeatures_37
  #llim23=llim22+nfeatures_22
  #llim24=llim23+nfeatures_23
  #llim25=llim24+nfeatures_24

  llim=[llim1, llim2, llim3, llim4, llim5, llim6, llim7, llim8, llim9, llim10, llim11, llim12, llim13, llim14, llim15, llim16, llim17, llim18,llim19,llim20,llim21,llim22,llim23,llim24,llim25,llim26,llim27,llim28,llim29,llim30,llim31,llim32,llim33,llim34,llim35,llim36,llim37]
  nfeatures=[nfeatures_1, nfeatures_2,nfeatures_3,nfeatures_4,nfeatures_5,nfeatures_6,nfeatures_7,nfeatures_8,nfeatures_9,nfeatures_10,nfeatures_11,nfeatures_12,nfeatures_13,nfeatures_14,nfeatures_15,nfeatures_16,nfeatures_17,nfeatures_18,nfeatures_19,nfeatures_20,nfeatures_21,nfeatures_22,nfeatures_23,nfeatures_24,nfeatures_25,nfeatures_26,nfeatures_27,nfeatures_28,nfeatures_29,nfeatures_30,nfeatures_31,nfeatures_32,nfeatures_33,nfeatures_34,nfeatures_35,nfeatures_36]


  for i, lf in enumerate(l_feat):
    print("trial"+str(lf))
    if(i==0):
      nump=np.arange(llim[lf], llim[lf]+nfeatures[lf])
      #=[]
    else:
      numpu=np.arange(llim[lf], llim[lf]+nfeatures[lf])
      nump=np.append(nump, numpu)
  numpl=nump.tolist()
  numpl = list(map(int,numpl))

  print(final.shape)

  final=final[:,numpl]
  #final_t=final_t[:,numpl]

  print(final.shape)

  list_rand=[]
  for i in range(0,final.shape[1]):
    list_rand.append("c"+str(i))
  len(list_rand)
  df_train = pd.DataFrame(final, columns = list_rand)
  #df_test = pd.DataFrame(final_t, columns = list_rand)

  return df_train, llim, nfeatures

