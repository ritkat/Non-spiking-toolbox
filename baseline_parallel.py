import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import*
from evaluate_encoder import*
from args_final import args as my_args
from itertools import product
from itertools import product, islice
import time

if __name__ == '__main__':

    args = my_args()
    #Run baseline mode for all features and all electrodes
    args.method="topn_elec"
    if(args.method=="genetic"):
        print(args.__dict__)
        # Fix the seed of all random number generator
        seed = 50
        random.seed(seed)
        np.random.seed(seed)
        df = pd.DataFrame({"dataset":[],"tstep":[], "accuracy":[], "accuracy std":[],"generation":[], "max_features":[],"gen accuracy":[], "gen std":[],"selected features":[], "nfeatures":[], "params":[]})

        parameters = dict(dataset=["bci3","jc_mot","fp_im", "jc_im", "jm_im", "rr_im", "rh_im", "bp_im","wc_mot","zt_mot","fp_mot","gc_mot","hh_mot","hl_mot","jf_mot","jp_mot","rh_mot","rr_mot","ug_mot","jt_mot","jm_mot","gf_mot","bp_mot","cc_mot","ca_mot","de_mot"],
        tstep=[500,1000,1500,3000], maxft=[None], classifier=["RF","SVM"]
        )

        n_generations=10
        l_feat=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        args.gen=n_generations
        args.l_feat=l_feat
        param_values = [v for v in parameters.values()]
        all_combinations = product(*param_values)
        iterator = islice(all_combinations, args.run, args.run+1)
        args.dataset,args.tstep,args.max_feat,args.classifier = next(iterator)
        
        accd, gen, self, nfeat, sd, genstd, params=genetic(args)
        for n in range(args.gen+1):
            df = df.append({"dataset":args.dataset,"tstep":args.tstep,"accuracy":accd, "accuracy std":sd,"generation":n, "max_features":args.max_feat,"gen accuracy":gen[str(n)],"gen std":genstd[str(n)],"selected features":self,"nfeatures":nfeat, "params":params},ignore_index=True)
            log_file_name = 'accuracy_log_'+str(args.run)+'.csv'
            pwd = os.getcwd()
            log_dir = pwd+'/log_dir/'
            df.to_csv(log_dir+log_file_name, index=False)

            df.to_csv(log_file_name, index=False)


    elif(args.method=="individual"):
        df = pd.DataFrame({"dataset":[],"tstep":[], "accuracy ind electrodes":[],"sd ind electrodes":[],"accuracy ind features":[],"sd ind features":[],"classifier":[], "best_params":[]})

        parameters = dict(dataset=["bci3","jc_mot","fp_im", "jc_im", "jm_im", "rr_im", "rh_im", "bp_im","wc_mot","zt_mot","fp_mot","gc_mot","hh_mot","hl_mot","jf_mot","jp_mot","rh_mot","rr_mot","ug_mot","jt_mot","jm_mot","gf_mot","bp_mot","cc_mot","ca_mot","de_mot"],
        tstep=[3000],classifier=["SVM","RF"]
        )

        l_feat=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        args.l_feat=l_feat

        param_values = [v for v in parameters.values()]

        all_combinations = product(*param_values)
        iterator = islice(all_combinations, args.run, args.run+1)
        args.dataset,args.tstep,args.classifier = next(iterator)

        acc,sd,accf,sdf,best_params=individual(args)
        #for n in range(args.gen+1):
        df = df.append({"dataset":args.dataset,"tstep":args.tstep,"accuracy ind electrodes":acc,"sd ind electrodes":sd,"accuracy ind features":accf,"sd ind features":sdf,"classifier":args.classifier,"best_params":best_params},ignore_index=True)
        log_file_name = 'accuracy_log_'+str(args.run)+'.csv'
        pwd = os.getcwd()
        log_dir = pwd+'/log_dir/'
        df.to_csv(log_dir+log_file_name, index=False)

        df.to_csv(log_file_name, index=False)

    elif(args.method=="baseline"):
        df = pd.DataFrame({"dataset":[],"tstep":[],"accuracy":[],"classifier":[],"best_params":[]})

        parameters = dict(dataset=["bci3","jc_mot","fp_im", "jc_im", "jm_im", "rr_im", "rh_im", "bp_im","wc_mot","zt_mot","fp_mot","gc_mot","hh_mot","hl_mot","jf_mot","jp_mot","rh_mot","rr_mot","ug_mot","jt_mot","jm_mot","gf_mot","bp_mot","cc_mot","ca_mot","de_mot"],
        tstep=[500,1000,1500,3000],classifier=["SVM","RF"]
        )

        n_generations=10
        l_feat=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        #args.gen=n_generations
        args.l_feat=l_feat
    
        param_values = [v for v in parameters.values()]
        all_combinations = product(*param_values)
        iterator = islice(all_combinations, args.run, args.run+1)
        args.dataset,args.tstep, args.classifier = next(iterator)

        accd,sd,best_params=baseline(args)

        df = df.append({"dataset":args.dataset,"tstep":args.tstep,"accuracy":accd, "std":sd,"classifier":args.classifier, "best_params":best_params},ignore_index=True)
        log_file_name = 'accuracy_log_'+str(args.run)+'.csv'
        
        pwd = os.getcwd()
        log_dir = pwd+'/log_dir/'
        df.to_csv(log_dir+log_file_name, index=False)

        df.to_csv(log_file_name, index=False)


    elif(args.method=="topn_elec"):
        df = pd.DataFrame({"dataset":[],"tstep":[],"accuracy":[],"std":[],"classifier":[],"best_params":[]})

        parameters = dict(dataset=["bci3","jc_mot","fp_im", "jc_im", "jm_im", "rr_im", "rh_im", "bp_im","wc_mot","zt_mot","fp_mot","gc_mot","hh_mot","hl_mot","jf_mot","jp_mot","rh_mot","rr_mot","ug_mot","jt_mot","jm_mot","gf_mot","bp_mot","cc_mot","ca_mot","de_mot"],
        tstep=[3000],classifier=["SVM","RF"]
        )
        n_generations=10
        l_feat=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        args.l_feat=l_feat

        param_values = [v for v in parameters.values()]
        all_combinations = product(*param_values)
        iterator = islice(all_combinations, args.run-100, args.run+1-100)
        args.dataset,args.tstep,args.classifier = next(iterator)
 
        acc,best_params,sd=topn_elec(args)
        df = df.append({"dataset":args.dataset,"tstep":args.tstep,"accuracy":acc, "std":sd,"classifier":args.classifier, "best_params":best_params},ignore_index=True)
        log_file_name = 'accuracy_topn_log_'+str(args.run)+'.csv'
        pwd = os.getcwd()
        log_dir = pwd+'/log_dir/'
        df.to_csv(log_dir+log_file_name, index=False)

        df.to_csv(log_file_name, index=False)