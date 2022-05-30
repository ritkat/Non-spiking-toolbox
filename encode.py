#!/usr/bin/env python
# coding: utf-8

# spike conversion algorithm.
# Spike time array contains values of spike times in ms.
# Saved arrays :

# X: Array of the EMG/EEG/ECoG Digital time series data with length = 200
# Y: Array of the labels of theing data with length = 200

# spike_times_up: Spike time arrays with upward polarity in ms for X. length = 200
# spike_times_dn: Spike time arrays with downward polarity in ms for X. length = 200

# Author : Nikhil Garg, 3IT Sherbrooke ; nikhilgarg.bits@gmail.com
# Created : 15 July 2020
# Last edited : 3rd January 2022

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.signal import butter, lfilter, welch, square  # for signal filtering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from utilis import *
from createFV import *
from args import args as my_args

def encode(args):
    # general stuff
    # sampling frequency of MYO
    
    VERBOSE = True
    pwd = os. getcwd()

    if args.dataset == "bci3":
        data_dir = pwd + "/dataset/bci3.npz"
        fs = 1000
        nb_channels = 64
        
    elif args.dataset =="zt_mot":
      data_dir = pwd + "/dataset/zt_mot_epochs.npz"
      nb_channels=48
      fs = 1000

    elif args.dataset =="jc_mot":
      data_dir = pwd + "/dataset/jc_mot_epochs.npz"
      nb_channels=48
      fs = 1000

    elif args.dataset =="fp_mot":
      data_dir = pwd + "/dataset/fp_mot_epochs.npz"
      nb_channels=62
      fs = 1000

    elif args.dataset =="ca_mot":
        data_dir = pwd + "/dataset/ca_mot_epochs.npz"
        nb_channels=59
        fs = 1000
        
    elif args.dataset =="jp_mot":
        data_dir = pwd + "/dataset/jp_mot_epochs.npz"
        nb_channels=58
        fs = 1000

    elif args.dataset =="jm_mot":
        data_dir = pwd + "/dataset/jm_mot_epochs.npz"
        nb_channels=63
        fs = 1000

    elif args.dataset =="hh_mot":
        data_dir = pwd + "/dataset/hh_mot_epochs.npz"
        nb_channels=41
        fs = 1000   

    elif args.dataset =="hl_mot":
        data_dir = pwd + "/dataset/hl_mot_epochs.npz"
        nb_channels=64
        fs = 1000  

    elif args.dataset =="gc_mot":
        data_dir = pwd + "/dataset/gc_mot_epochs.npz"
        nb_channels=64
        fs = 1000  

    elif args.dataset =="ug_mot":
        data_dir = pwd + "/dataset/ug_mot_epochs.npz"
        nb_channels=25
        fs = 1000  

    elif args.dataset =="wc_mot":
        data_dir = pwd + "/dataset/wc_mot_epochs.npz"
        nb_channels=64
        fs = 1000


    elif args.dataset =="jf_mot":
        data_dir = pwd + "/dataset/jf_mot_epochs.npz"
        nb_channels=39
        fs = 1000

    elif args.dataset =="bp_mot":
        data_dir = pwd + "/dataset/bp_mot_epochs.npz"
        nb_channels=47
        fs = 1000

    elif args.dataset =="de_mot":
        data_dir = pwd + "/dataset/de_mot_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.dataset =="cc_mot":
        data_dir = pwd + "/dataset/cc_mot_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.dataset =="rr_mot":
        data_dir = pwd + "/dataset/rr_mot_epochs.npz"
        nb_channels=49
        fs = 1000

    elif args.dataset =="jt_mot":
        data_dir = pwd + "/dataset/jt_mot_epochs.npz"
        nb_channels=62
        fs = 1000

    elif args.dataset =="gf_mot":
        data_dir = pwd + "/dataset/gf_mot_epochs.npz"
        nb_channels=63
        fs = 1000

    elif args.dataset =="rh_mot":
        data_dir = pwd + "/dataset/rh_mot_epochs.npz"
        nb_channels=63
        fs = 1000

    elif args.dataset =="rr_im":
        data_dir = pwd + "/dataset/rr_im_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.dataset =="jc_im":
        data_dir = pwd + "/dataset/jc_im_epochs.npz"
        nb_channels=48
        fs = 1000

    elif args.dataset =="jm_im":
        data_dir = pwd + "/dataset/jm_im_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.dataset =="fp_im":
        data_dir = pwd + "/dataset/fp_im_epochs.npz"
        nb_channels=64
        fs = 1000

    elif args.dataset =="bp_im":
        data_dir = pwd + "/dataset/bp_im_epochs.npz"
        nb_channels=46
        fs = 1000

    elif args.dataset =="rh_im":
        data_dir = pwd + "/dataset/rh_im_epochs.npz"
        nb_channels=64
        fs = 1000

    spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, X_Train_list,X_Test_list, Y_Train_list,Y_Test_list,avg_spike_rate_list=[], [],[], [],[], [],[], [],[]

    if args.dataset=="bci3":
        #Add data here
        X_Train = []
        Y_Train = []
        X_Test = []
        Y_Test = []

        data = np.load(data_dir)
        X_Train = data['X_Train']
        Y_Train = data['Y_Train']

        X_Test = data['X_Test']
        Y_Test = data['Y_Test']

        X_Train_list.append(X_Train)
        Y_Train_list.append(Y_Train)
        X_Test_list.append(X_Test)
        Y_Test_list.append(Y_Test)



        X_Train = np.array(X_Train)

        #X_Train = np.moveaxis(X_Train, 2, 1)
        Y_Train = np.array(Y_Train)

        X_Test = np.array(X_Test)
        #X_Test = np.moveaxis(X_Test, 2, 1)
        Y_Test = np.array(Y_Test)

        if(args.preprocess==1):
        #subsampling by 4 
            data_2_subs=X_Train
            '''data_2_subs=np.zeros((data_train.shape[0], data_train.shape[1], int(data_train.shape[2]/4)))
            for i in range(0, data_train.shape[0]):
                for j in range(0, data_train.shape[1]):
                    data_2_subs[i, j, :]=signal.resample(data_2_sub[i, j, :], int(data_train.shape[2]/4))'''

            #data_2_subs.shape
            
            
            for j in range(0, X_Train.shape[0]):
                car=np.zeros((data_2_subs.shape[2],))
                for i in range(0, X_Train.shape[1]):
                    car= car + data_2_subs[j,i,:]
                car=car/X_Train.shape[1]
                #car.shape
                for k in range(0, X_Train.shape[1]):
                    data_2_subs[j,k,:]=data_2_subs[j,k,:]-car
                        
            #subsampling by 4 
            data_2_subs_t=X_Test
            '''data_2_subs_t=np.zeros((data_test.shape[0], data_test.shape[1], int(data_test.shape[2]/4)))
            for i in range(0, data_test.shape[0]):
                for j in range(0, data_test.shape[1]):
                    data_2_subs_t[i, j, :]=signal.resample(data_2_sub_t[i, j, :], int(data_test.shape[2]/4))'''

            #data_2_subs_t.shape
            #Common Average Reference
            #if(c_ref==True):
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



        # X_uniform is a time series data array with length of 400. The initial segments are about 397, 493 etc which
        # makes it incompatible in some cases where uniform input is desired.

        nb_trials = X_Train.shape[0]

            
        # print(len(X))
        print("Number of training samples in dataset:")
        print(len(X_Train))
        print(len(Y_Train))
        # print("Class labels:")
        # print(list(set(Y_Train)))

        # Take session 0,1 as and session 2 as test.


        interpfact = args.encode_interpfact
        refractory_period = args.encode_refractory  # in ms
        th_up = args.encode_thr_up
        th_dn = args.encode_thr_dn


        # Generate the  data
        X=X_Train
        Y=Y_Train
        spike_times_train_up = []
        spike_times_train_dn = []
        for i in range(len(X)):
            spk_up, spk_dn = gen_spike_time(
                time_series_data=X[i],
                interpfact=interpfact,
                fs=fs,
                th_up=th_up,
                th_dn=th_dn,
                refractory_period=refractory_period,
            )
            spike_times_train_up.append(spk_up)
            spike_times_train_dn.append(spk_dn)
        spike_times_train_up_list.append(spike_times_train_up)
        spike_times_train_dn_list.append(spike_times_train_dn)

        

        rate_up = gen_spike_rate(spike_times_train_up)
        rate_dn = gen_spike_rate(spike_times_train_dn)
        avg_spike_rate = (rate_up+rate_dn)/2
        print("Average spiking rate")
        print(avg_spike_rate)

            # Generate the  data
        X=X_Test
        Y=Y_Test
        spike_times_test_up = []
        spike_times_test_dn = []
        for i in range(len(X)):
            spk_up, spk_dn = gen_spike_time(
                time_series_data=X[i],
                interpfact=interpfact,
                fs=fs,
                th_up=th_up,
                th_dn=th_dn,
                refractory_period=refractory_period,
            )
            spike_times_test_up.append(spk_up)
            spike_times_test_dn.append(spk_dn)
        spike_times_test_up_list.append(spike_times_test_up)
        spike_times_test_dn_list.append(spike_times_test_dn)
        



        nb_trials = X_Test.shape[0]

            
        # print(len(X))
        print("Number of test samples in dataset:")
        print(len(X_Test))
        print(len(Y_Test))
        # print("Class labels:")
        # print(list(set(Y_Test)))


        spike_times_train_up = np.array(spike_times_train_up)
        spike_times_test_up = np.array(spike_times_test_up)
        spike_times_train_dn = np.array(spike_times_train_dn)
        spike_times_test_dn = np.array(spike_times_test_dn)


        file_path = "dataset/"
        file_name = args.encoded_data_file_prefix + str(args.dataset) + str(args.encode_thr_up) + str(
            args.encode_thr_dn) + str(args.encode_refractory) + str(args.encode_interpfact) + ".npz"

        np.savez_compressed(
            file_path + file_name,
            # X_Train=X_Train,
            Y_Train=Y_Train,
            # X_Test=X_Test,
            Y_Test=Y_Test,
            spike_times_train_up=spike_times_train_up,
            spike_times_train_dn=spike_times_train_dn,
            spike_times_test_up=spike_times_test_up,
            spike_times_test_dn=spike_times_test_dn,
        )
        return spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, X_Train_list,X_Test_list, Y_Train_list,Y_Test_list,avg_spike_rate_list

    else:
        #Add data here
        X_Train = []
        Y_Train = []
        X_Test = []
        Y_Test = []

        data = np.load(data_dir)
        X_Train = data['X']
        Y_Train = data['y']

        kf3 = KFold(n_splits=args.kfold, shuffle=False)
        
        for train_index, test_index in kf3.split(X_Train):
            X_Tr = X_Train[train_index]
            Y_Tr = Y_Train[train_index]
            X_Test = X_Train[test_index]
            Y_Test = Y_Train[test_index]
            X_Train_list.append(X_Tr)
            Y_Train_list.append(Y_Tr)
            X_Test_list.append(X_Test)
            Y_Test_list.append(Y_Test)



            if(args.preprocess==1):
                data_2_subs=X_Tr
                '''data_2_subs=np.zeros((data_train.shape[0], data_train.shape[1], int(data_train.shape[2]/4)))
                for i in range(0, data_train.shape[0]):
                    for j in range(0, data_train.shape[1]):
                        data_2_subs[i, j, :]=signal.resample(data_2_sub[i, j, :], int(data_train.shape[2]/4))'''

                #data_2_subs.shape
                for j in range(0, X_Tr.shape[0]):
                    car=np.zeros((data_2_subs.shape[2],))
                    for i in range(0, X_Tr.shape[1]):
                        car= car + data_2_subs[j,i,:]
                    car=car/X_Tr.shape[1]
                    #car.shape
                    for k in range(0, X_Tr.shape[1]):
                        data_2_subs[j,k,:]=data_2_subs[j,k,:]-car
                            
                #subsampling by 4 
                data_2_subs_t=X_Test
                '''data_2_subs_t=np.zeros((data_test.shape[0], data_test.shape[1], int(data_test.shape[2]/4)))
                for i in range(0, data_test.shape[0]):
                    for j in range(0, data_test.shape[1]):
                        data_2_subs_t[i, j, :]=signal.resample(data_2_sub_t[i, j, :], int(data_test.shape[2]/4))'''

                #data_2_subs_t.shape
                #Common Average Reference
                #if(c_ref==True):
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



            # X_uniform is a time series data array with length of 400. The initial segments are about 397, 493 etc which
            # makes it incompatible in some cases where uniform input is desired.

            nb_trials = X_Tr.shape[0]

                
            # print(len(X))
            print("Number of training samples in dataset:")
            print(len(X_Tr))
            print(len(Y_Tr))
            # print("Class labels:")
            # print(list(set(Y_Train)))

            # Take session 0,1 as and session 2 as test.


            interpfact = args.encode_interpfact
            refractory_period = args.encode_refractory  # in ms
            th_up = args.encode_thr_up
            th_dn = args.encode_thr_dn


            # Generate the  data
            X=X_Tr
            Y=Y_Tr
            spike_times_train_up = []
            spike_times_train_dn = []
            for i in range(len(X)):
                spk_up, spk_dn = gen_spike_time(
                    time_series_data=X[i],
                    interpfact=interpfact,
                    fs=fs,
                    th_up=th_up,
                    th_dn=th_dn,
                    refractory_period=refractory_period,
                )
                spike_times_train_up.append(spk_up)
                spike_times_train_dn.append(spk_dn)
            spike_times_train_up_list.append(spike_times_train_up)
            spike_times_train_dn_list.append(spike_times_train_dn)

            rate_up = gen_spike_rate(spike_times_train_up)
            rate_dn = gen_spike_rate(spike_times_train_dn)
            avg_spike_rate = (rate_up+rate_dn)/2
            print("Average spiking rate")
            print(avg_spike_rate)
            avg_spike_rate_list.append(avg_spike_rate)

                # Generate the  data
            X=X_Test
            Y=Y_Test
            spike_times_test_up = []
            spike_times_test_dn = []
            for i in range(len(X)):
                spk_up, spk_dn = gen_spike_time(
                    time_series_data=X[i],
                    interpfact=interpfact,
                    fs=fs,
                    th_up=th_up,
                    th_dn=th_dn,
                    refractory_period=refractory_period,
                )
                spike_times_test_up.append(spk_up)
                spike_times_test_dn.append(spk_dn)
            spike_times_test_up_list.append(spike_times_test_up)
            spike_times_test_dn_list.append(spike_times_test_dn)



            nb_trials = X_Test.shape[0]

                
            # print(len(X))
            print("Number of test samples in dataset:")
            print(len(X_Test))
            print(len(Y_Test))
            # print("Class labels:")
            # print(list(set(Y_Test)))


            spike_times_train_up = np.array(spike_times_train_up)
            spike_times_test_up = np.array(spike_times_test_up)
            spike_times_train_dn = np.array(spike_times_train_dn)
            spike_times_test_dn = np.array(spike_times_test_dn)


            file_path = "dataset/"
            file_name = args.encoded_data_file_prefix + str(args.dataset) + str(args.encode_thr_up) + str(
                args.encode_thr_dn) + str(args.encode_refractory) + str(args.encode_interpfact) + ".npz"

            np.savez_compressed(
                file_path + file_name,
                # X_Train=X_Train,
                Y_Train=Y_Tr,
                # X_Test=X_Test,
                Y_Test=Y_Test,
                spike_times_train_up=spike_times_train_up,
                spike_times_train_dn=spike_times_train_dn,
                spike_times_test_up=spike_times_test_up,
                spike_times_test_dn=spike_times_test_dn,
            )
        return spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, X_Train_list,X_Test_list, Y_Train_list,Y_Test_list,avg_spike_rate_list


if __name__ == '__main__':
    args = my_args()
    print(args.__dict__)
    # Fix the seed of all random number generator
    encode(args)
