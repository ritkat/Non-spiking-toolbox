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
from tqdm import tqdm
from args_final import args as my_args

args=my_args()

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

def createFV_individual(data_train, data_test, fs, l_feat, c_ref):

  #subsampling by 4 
  
  data_2_subs=data_train
  '''data_2_subs=np.zeros((data_train.shape[0], data_train.shape[1], int(data_train.shape[2]/4)))
  for i in range(0, data_train.shape[0]):
      for j in range(0, data_train.shape[1]):
          data_2_subs[i, j, :]=signal.resample(data_2_sub[i, j, :], int(data_train.shape[2]/4))'''

  #data_2_subs.shape
  
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

  #Standard Scaler

  '''for j in range(0, data_train.shape[0]):
      kr=data_2_subs[j,:,:]
      kr=data_2_subs[j,:,:]
      
      scaler=StandardScaler().fit(kr.T)
      data_2_subs[j,:,:]=scaler.transform(kr.T).T'''
      
  for i in tqdm(range(data_train.shape[0])):
    data_train_trial=data_2_subs[i,:,:]
    if(i==0):
      data_new=data_train_trial
    else:
      data_new=np.hstack((data_new, data_train_trial))
      
  for i in tqdm(range(data_test.shape[0])):
    data_train_trial_t=data_2_subs_t[i,:,:]
    if(i==0):
      data_new_t=data_train_trial_t
    else:
      data_new_t=np.hstack((data_new_t, data_train_trial_t))
      
  scaler = StandardScaler()
  #param_ls=[]
  for j in tqdm(range(data_new.shape[0])):
    scaler.fit(data_new[j,np.newaxis,:])
    data_new[j,:]=scaler.transform(data_new[j,np.newaxis,:])
    data_new_t[j,:]=scaler.transform(data_new_t[j,np.newaxis,:])
    
  data_2_subs=np.reshape(data_new, (data_train.shape[0], data_train.shape[1], data_train.shape[2]))
  data_2_subs_t=np.reshape(data_new_t, (data_test.shape[0], data_test.shape[1], data_test.shape[2]))
  

  '''#bandpass filter
  b, a = signal.butter(2, 0.4, 'low', analog=False)
  data_2_subs = signal.filtfilt(b, a, data_2_subs, axis=2)'''

  #Extracting all the features and concatenating them 
  final = np.array([])
  for j in range(0, data_2_subs.shape[0]):
      data_trial=data_2_subs[j,:,:].T
      #data_trial.shape

      data_trial_s1=data_trial[0:int(data_2_subs.shape[2]/3),:]
      #print(data_trial_s1.shape)
      data_trial_s2=data_trial[int(data_2_subs.shape[2]/3):2*int(data_2_subs.shape[2]/3),:]
      #print(data_trial_s2.shape)
      data_trial_s3=data_trial[2*int(data_2_subs.shape[2]/3):3*int(data_2_subs.shape[2]/3),:]
      #print(data_trial_s3.shape)

      #AR Coefficients

      #from statsmodels.datasets.sunspots import load
      #data = load()
      ARFV=np.array([])

      for i in range(0, data_train.shape[1]):
          rho1, sigma1 = sm.regression.linear_model.burg(data_trial_s1[:,i], order=2)
          rho2, sigma2 = sm.regression.linear_model.burg(data_trial_s2[:,i], order=2)
          rho3, sigma3 = sm.regression.linear_model.burg(data_trial_s3[:,i], order=2)
          ARFV=np.append(ARFV, (rho1, rho2, rho3))

      #print(ARFV) 

      #Haar wavelet

      HWDFV=np.array([])
      for i in range(0, data_train.shape[1]):
          (cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          HWDFV=np.append(HWDFV, cA)

      #Spectral Power estimates
      SPFV=np.array([])
      for i in range(0, data_train.shape[1]):
          f1, Pxx_den1 = signal.welch(data_trial_s1[:,i], int(data_2_subs.shape[2]/3))
          f2, Pxx_den2 = signal.welch(data_trial_s2[:,i], int(data_2_subs.shape[2]/3))
          f3, Pxx_den3 = signal.welch(data_trial_s3[:,i], int(data_2_subs.shape[2]/3))
          SPFV=np.append(SPFV, (Pxx_den1, Pxx_den2, Pxx_den3))

      #Concatenaton of All the feature vectors
      concated=np.concatenate((ARFV, HWDFV, SPFV), axis=None)
      concated=np.reshape(concated, (-1, 1))
      if j==0:
          final=concated
      else:
          final= np.hstack((final, concated))
      print(j)
  print(final.shape)

  final=final.T

  eegData=np.rollaxis(data_2_subs, 0, 3)
  eegData.shape



  # Subband Information Quantity
  # delta (0.5–4 Hz)
  eegData_delta = eeg.filt_data(eegData, 0.5, 4, fs)
  ShannonRes_delta = eeg.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2)
  # theta (4–8 Hz)
  eegData_theta = eeg.filt_data(eegData, 4, 8, fs)
  ShannonRes_theta = eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2)
  # alpha (8–12 Hz)
  eegData_alpha = eeg.filt_data(eegData, 8, 12, fs)
  ShannonRes_alpha = eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2)
  # beta (12–30 Hz)
  eegData_beta = eeg.filt_data(eegData, 12, 30, fs)
  ShannonRes_beta = eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2)
  # gamma (30–100 Hz)
  eegData_gamma = eeg.filt_data(eegData, 30, 80, fs)
  ShannonRes_gamma = eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2)

  # Hjorth Mobility
  # Hjorth Complexity
  HjorthMob, HjorthComp = eeg.hjorthParameters(eegData)


  # Median Frequency
  medianFreqRes = eeg.medianFreq(eegData,fs)

  # Standard Deviation
  std_res = eeg.eegStd(eegData)

  # Regularity (burst-suppression)
  regularity_res = eeg.eegRegularity(eegData,fs)

  # Spikes
  minNumSamples = int(70*fs/1000)
  spikeNum_res = eeg.spikeNum(eegData,minNumSamples)

  # Sharp spike
  sharpSpike_res = eeg.shortSpikeNum(eegData,minNumSamples)
  
  #remove from here REMPOMOMOPMPOM
  concated=np.concatenate((ShannonRes_delta.T, ShannonRes_theta.T, ShannonRes_alpha.T, ShannonRes_beta.T, ShannonRes_gamma.T, HjorthMob.T, HjorthComp.T, medianFreqRes.T, std_res.T, regularity_res.T, spikeNum_res.T, sharpSpike_res.T), axis=1)

  final=np.hstack((final, concated))

  # δ band Power
  bandPwr_delta = eeg.bandPower(eegData, 0.5, 4, fs)
  # θ band Power
  bandPwr_theta = eeg.bandPower(eegData, 4, 8, fs)
  # α band Power
  bandPwr_alpha = eeg.bandPower(eegData, 8, 12, fs)
  # β band Power
  bandPwr_beta = eeg.bandPower(eegData, 12, 30, fs)
  # γ band Power
  bandPwr_gamma = eeg.bandPower(eegData, 30, 80, fs)

  concated_n=bandPwr_gamma.T
  final=np.hstack((final, concated_n))

  

  HTFV=np.array([])
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
  final.shape

  for j in range(0, data_2_subs_t.shape[0]):
      kr=data_2_subs_t[j,:,:]
      
      scaler=StandardScaler().fit(kr.T)
      data_2_subs_t[j,:,:]=scaler.transform(kr.T).T

  '''#bandpass filter
  b, a = signal.butter(2, 0.4, 'low', analog=False)
  data_2_subs_t = signal.filtfilt(b, a, data_2_subs_t, axis=2)'''

  final_t = np.array([])
  for j in range(0 ,data_2_subs_t.shape[0]):
      data_trial=data_2_subs_t[j,:,:].T
      #data_trial.shape

      data_trial_s1=data_trial[0:int(data_2_subs_t.shape[2]/3),:]
      #print(data_trial_s1.shape)
      data_trial_s2=data_trial[int(data_2_subs_t.shape[2]/3):2*int(data_2_subs_t.shape[2]/3),:]
      #print(data_trial_s2.shape)
      data_trial_s3=data_trial[2*int(data_2_subs_t.shape[2]/3):3*int(data_2_subs_t.shape[2]/3),:]
      #print(data_trial_s3.shape)

      #AR Coefficients
      #from statsmodels.datasets.sunspots import load
      #data = load()
      ARFV=np.array([])

      for i in range(0, data_2_subs_t.shape[1]):
          rho1, sigma1 = sm.regression.linear_model.burg(data_trial_s1[:,i], order=2)
          rho2, sigma2 = sm.regression.linear_model.burg(data_trial_s2[:,i], order=2)
          rho3, sigma3 = sm.regression.linear_model.burg(data_trial_s3[:,i], order=2)
          ARFV=np.append(ARFV, (rho1, rho2, rho3))

      #print(ARFV) 

      HWDFV=np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
          (cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          HWDFV=np.append(HWDFV, cA)

      #Spectral Power estimates
      SPFV=np.array([])
      for i in range(0, data_2_subs_t.shape[1]):
          f1, Pxx_den1 = signal.welch(data_trial_s1[:,i], int(data_2_subs_t.shape[2]/3))
          f2, Pxx_den2 = signal.welch(data_trial_s2[:,i], int(data_2_subs_t.shape[2]/3))
          f3, Pxx_den3 = signal.welch(data_trial_s3[:,i], int(data_2_subs_t.shape[2]/3))
          SPFV=np.append(SPFV, (Pxx_den1, Pxx_den2, Pxx_den3))

      #Concatenaton of All the feature vectors
      concated=np.concatenate((ARFV, HWDFV, SPFV), axis=None)
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
  eegData_t.shape


  # Subband Information Quantity
  # delta (0.5–4 Hz)
  eegData_delta_t = eeg.filt_data(eegData_t, 0.5, 4, fs)
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
  ShannonRes_gamma_t = eeg.shannonEntropy(eegData_gamma_t, bin_min=-200, bin_max=200, binWidth=2)

  # Hjorth Mobility
  # Hjorth Complexity
  HjorthMob_t, HjorthComp_t = eeg.hjorthParameters(eegData_t)


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
  sharpSpike_res_t = eeg.shortSpikeNum(eegData_t,minNumSamples)
  #REMPMVPOMMPOMPOMPOMPOMOMPOMPOM
  concated_t=np.concatenate(( ShannonRes_delta_t.T, ShannonRes_theta_t.T, ShannonRes_alpha_t.T, ShannonRes_beta_t.T, ShannonRes_gamma_t.T, HjorthMob_t.T, HjorthComp_t.T, medianFreqRes_t.T, std_res_t.T, regularity_res_t.T, spikeNum_res_t.T, sharpSpike_res_t.T), axis=1)

  final_t=np.hstack((final_t, concated_t))

  # δ band Power
  bandPwr_delta_t = eeg.bandPower(eegData_t, 0.5, 4, fs)
  #too large
  # θ band Power
  bandPwr_theta_t = eeg.bandPower(eegData_t, 4, 8, fs)
  #too large
  # α band Power
  bandPwr_alpha_t = eeg.bandPower(eegData_t, 8, 12, fs)
  #too large
  # β band Power
  bandPwr_beta_t = eeg.bandPower(eegData_t, 12, 30, fs)
  #too large
  # γ band Power
  bandPwr_gamma_t = eeg.bandPower(eegData_t, 30, 80, fs)

  concated_n_t= bandPwr_gamma_t.T
  final_t=np.hstack((final_t, concated_n_t))
  #final_t.shape

  HTFV_t=np.array([])
  for j in range(0, eegData_t.shape[2]):
    eegData_temp=eegData_t[:,:,j]
    HTFV_temp=np.array([])
    for i in range(0, eegData.shape[0]):
      HTFV_temp=np.append(HTFV_temp, np.imag(hilbert(eegData_temp[i,:])))
    if(j==0):
      HTFV_t=HTFV_temp
    else:
      HTFV_t=np.vstack((HTFV_t, HTFV_temp))


    print(j)

  final_t=np.hstack((final_t, HTFV_t))
  final_t.shape

  #importance per feature
  nfeatures_1=ARFV.shape[0]
  nfeatures_2=HWDFV.shape[0]
  nfeatures_3=SPFV.shape[0]

  #EEG EXTRACT FEATURES
  nfeatures_4=ShannonRes_delta.shape[0]
  nfeatures_5=ShannonRes_theta.shape[0]
  nfeatures_6=ShannonRes_alpha.shape[0]
  nfeatures_7=ShannonRes_beta.shape[0]
  nfeatures_8=ShannonRes_gamma.shape[0]
  nfeatures_9=HjorthMob.shape[0]
  nfeatures_10=HjorthComp.shape[0]
  nfeatures_11=medianFreqRes.shape[0]
  nfeatures_12=std_res.shape[0]
  nfeatures_13=regularity_res.shape[0]
  nfeatures_14=spikeNum_res.shape[0]
  nfeatures_15=sharpSpike_res.shape[0]
  nfeatures_16=bandPwr_gamma.shape[0]
  nfeatures_17=HTFV_temp.shape[0]
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
  '''llim17=llim16+nfeatures_16
  llim18=llim17+nfeatures_17
  llim19=llim18+nfeatures_18
  llim20=llim19+nfeatures_19'''

  llim=[llim1, llim2, llim3, llim4, llim5, llim6, llim7, llim8, llim9, llim10, llim11, llim12, llim13, llim14, llim15, llim16, llim17, llim18]
  nfeatures=[nfeatures_1, nfeatures_2,nfeatures_3,nfeatures_4,nfeatures_5,nfeatures_6,nfeatures_7,nfeatures_8,nfeatures_9,nfeatures_10,nfeatures_11,nfeatures_12,nfeatures_13,nfeatures_14,nfeatures_15,nfeatures_16,nfeatures_17]

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

def createFV_individual_feat(data_train, fs, l_feat, c_ref):

  #subsampling by 4 
  
  data_2_subs=data_train
  '''data_2_subs=np.zeros((data_train.shape[0], data_train.shape[1], int(data_train.shape[2]/4)))
  for i in range(0, data_train.shape[0]):
      for j in range(0, data_train.shape[1]):
          data_2_subs[i, j, :]=signal.resample(data_2_sub[i, j, :], int(data_train.shape[2]/4))'''

  #data_2_subs.shape
  
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

  #Standard Scaler

  for j in range(0, data_train.shape[0]):
      kr=data_2_subs[j,:,:]
      kr=data_2_subs[j,:,:]
      
      scaler=StandardScaler().fit(kr.T)
      data_2_subs[j,:,:]=scaler.transform(kr.T).T

  '''#bandpass filter
  b, a = signal.butter(2, 0.4, 'low', analog=False)
  data_2_subs = signal.filtfilt(b, a, data_2_subs, axis=2)'''

  #Extracting all the features and concatenating them 
  final = np.array([])
  for j in range(0, data_2_subs.shape[0]):
      data_trial=data_2_subs[j,:,:].T
      #data_trial.shape

      data_trial_s1=data_trial[0:int(data_2_subs.shape[2]/3),:]
      #print(data_trial_s1.shape)
      data_trial_s2=data_trial[int(data_2_subs.shape[2]/3):2*int(data_2_subs.shape[2]/3),:]
      #print(data_trial_s2.shape)
      data_trial_s3=data_trial[2*int(data_2_subs.shape[2]/3):3*int(data_2_subs.shape[2]/3),:]
      #print(data_trial_s3.shape)

      #AR Coefficients

      #from statsmodels.datasets.sunspots import load
      #data = load()
      ARFV=np.array([])

      for i in range(0, data_train.shape[1]):
          rho1, sigma1 = sm.regression.linear_model.burg(data_trial_s1[:,i], order=2)
          rho2, sigma2 = sm.regression.linear_model.burg(data_trial_s2[:,i], order=2)
          rho3, sigma3 = sm.regression.linear_model.burg(data_trial_s3[:,i], order=2)
          ARFV=np.append(ARFV, (rho1, rho2, rho3))

      #print(ARFV) 

      #Haar wavelet

      HWDFV=np.array([])
      for i in range(0, data_train.shape[1]):
          (cA, cD) = pywt.dwt(data_trial[:,i], 'haar')
          HWDFV=np.append(HWDFV, cA)

      #Spectral Power estimates
      SPFV=np.array([])
      for i in range(0, data_train.shape[1]):
          f1, Pxx_den1 = signal.welch(data_trial_s1[:,i], int(data_2_subs.shape[2]/3))
          f2, Pxx_den2 = signal.welch(data_trial_s2[:,i], int(data_2_subs.shape[2]/3))
          f3, Pxx_den3 = signal.welch(data_trial_s3[:,i], int(data_2_subs.shape[2]/3))
          SPFV=np.append(SPFV, (Pxx_den1, Pxx_den2, Pxx_den3))

      #Concatenaton of All the feature vectors
      concated=np.concatenate((ARFV, HWDFV, SPFV), axis=None)
      concated=np.reshape(concated, (-1, 1))
      if j==0:
          final=concated
      else:
          final= np.hstack((final, concated))
      print(j)
  print(final.shape)

  final=final.T

  eegData=np.rollaxis(data_2_subs, 0, 3)
  eegData.shape



  # Subband Information Quantity
  # delta (0.5–4 Hz)
  eegData_delta = eeg.filt_data(eegData, 0.5, 4, fs)
  ShannonRes_delta = eeg.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2)
  # theta (4–8 Hz)
  eegData_theta = eeg.filt_data(eegData, 4, 8, fs)
  ShannonRes_theta = eeg.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2)
  # alpha (8–12 Hz)
  eegData_alpha = eeg.filt_data(eegData, 8, 12, fs)
  ShannonRes_alpha = eeg.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2)
  # beta (12–30 Hz)
  eegData_beta = eeg.filt_data(eegData, 12, 30, fs)
  ShannonRes_beta = eeg.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2)
  # gamma (30–100 Hz)
  eegData_gamma = eeg.filt_data(eegData, 30, 80, fs)
  ShannonRes_gamma = eeg.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2)

  # Hjorth Mobility
  # Hjorth Complexity
  HjorthMob, HjorthComp = eeg.hjorthParameters(eegData)


  # Median Frequency
  medianFreqRes = eeg.medianFreq(eegData,fs)

  # Standard Deviation
  std_res = eeg.eegStd(eegData)

  # Regularity (burst-suppression)
  regularity_res = eeg.eegRegularity(eegData,fs)

  # Spikes
  minNumSamples = int(70*fs/1000)
  spikeNum_res = eeg.spikeNum(eegData,minNumSamples)

  # Sharp spike
  sharpSpike_res = eeg.shortSpikeNum(eegData,minNumSamples)
  
  #remove from here REMPOMOMOPMPOM
  concated=np.concatenate((ShannonRes_delta.T, ShannonRes_theta.T, ShannonRes_alpha.T, ShannonRes_beta.T, ShannonRes_gamma.T, HjorthMob.T, HjorthComp.T, medianFreqRes.T, std_res.T, regularity_res.T, spikeNum_res.T, sharpSpike_res.T), axis=1)

  final=np.hstack((final, concated))

  # δ band Power
  bandPwr_delta = eeg.bandPower(eegData, 0.5, 4, fs)
  # θ band Power
  bandPwr_theta = eeg.bandPower(eegData, 4, 8, fs)
  # α band Power
  bandPwr_alpha = eeg.bandPower(eegData, 8, 12, fs)
  # β band Power
  bandPwr_beta = eeg.bandPower(eegData, 12, 30, fs)
  # γ band Power
  bandPwr_gamma = eeg.bandPower(eegData, 30, 80, fs)

  concated_n=bandPwr_gamma.T
  final=np.hstack((final, concated_n))

  

  HTFV=np.array([])
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
  final.shape



 

  #importance per feature
  nfeatures_1=ARFV.shape[0]
  nfeatures_2=HWDFV.shape[0]
  nfeatures_3=SPFV.shape[0]

  #EEG EXTRACT FEATURES
  nfeatures_4=ShannonRes_delta.shape[0]
  nfeatures_5=ShannonRes_theta.shape[0]
  nfeatures_6=ShannonRes_alpha.shape[0]
  nfeatures_7=ShannonRes_beta.shape[0]
  nfeatures_8=ShannonRes_gamma.shape[0]
  nfeatures_9=HjorthMob.shape[0]
  nfeatures_10=HjorthComp.shape[0]
  nfeatures_11=medianFreqRes.shape[0]
  nfeatures_12=std_res.shape[0]
  nfeatures_13=regularity_res.shape[0]
  nfeatures_14=spikeNum_res.shape[0]
  nfeatures_15=sharpSpike_res.shape[0]
  nfeatures_16=bandPwr_gamma.shape[0]
  nfeatures_17=HTFV_temp.shape[0]
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
  '''llim17=llim16+nfeatures_16
  llim18=llim17+nfeatures_17
  llim19=llim18+nfeatures_18
  llim20=llim19+nfeatures_19'''

  llim=[llim1, llim2, llim3, llim4, llim5, llim6, llim7, llim8, llim9, llim10, llim11, llim12, llim13, llim14, llim15, llim16, llim17, llim18]
  nfeatures=[nfeatures_1, nfeatures_2,nfeatures_3,nfeatures_4,nfeatures_5,nfeatures_6,nfeatures_7,nfeatures_8,nfeatures_9,nfeatures_10,nfeatures_11,nfeatures_12,nfeatures_13,nfeatures_14,nfeatures_15,nfeatures_16,nfeatures_17]

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

