import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import*
from createFV import *
from functions import *
from evaluate_encoder import*
from preprocessed import *
#from evaluate_reservoir import *
#from utilis import *
from args_final import args as my_args
from itertools import product
from itertools import product, islice
import time
from preprocess_individual import *

if __name__ == '__main__':

    args = my_args()
    #Run baseline mode for all features and all electrodes
    args.method="baseline"
    if(args.method=="genetic"):
        print(args.__dict__)
        # Fix the seed of all random number generator
        seed = int(args.seed)
        seed = 50
        random.seed(seed)
        np.random.seed(seed)
        df = pd.DataFrame({"dataset":[],"f_split":[],"l_feat":[],"tstep":[], "accuracy":[], "accuracy std":[],"generation":[], "max_features":[],"gen accuracy":[], "gen std":[],"selected features":[], "nfeatures":[], "params":[]})

        parameters = dict(dataset=["jc_mot","fp_im", "jc_im", "jm_im", "rr_im", "rh_im", "bp_im","wc_mot","zt_mot","fp_mot","gc_mot","hh_mot","hl_mot","jf_mot","jp_mot","rh_mot","rr_mot","ug_mot","jt_mot","jm_mot","gf_mot","bp_mot","cc_mot","ca_mot","de_mot"],
        tstep=[500, 1000], maxft=[5], classifier=["RF"],f_split=[2,3],l_feat=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]]
        )

        n_generations=2
        #l_feat=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
        args.gen=n_generations
        #args.l_feat=l_feat

        param_values = [v for v in parameters.values()]

        for args.dataset,args.tstep,args.max_feat,args.classifier,args.f_split,args.l_feat in product(*param_values):
            accd, gen, self, nfeat, sd, genstd, params=genetic(args)
            for n in range(args.gen+1):
                df = df.append({"dataset":args.dataset,"f_split":args.f_split,"l_feat":args.l_feat,"tstep":args.tstep,"accuracy":accd, "accuracy std":sd,"generation":n, "max_features":args.max_feat,"gen accuracy":gen[str(n)],"gen std":genstd[str(n)],"selected features":self,"nfeatures":nfeat, "params":params},ignore_index=True)
                log_file_name = 'accuracy_log_'+args.dataset+'.csv'
                pwd = os.getcwd()
                log_dir = pwd+'/log_dir/'
                df.to_csv(log_dir+log_file_name, index=False)

                df.to_csv(log_file_name, index=False)

            '''accuracy_df = pd.DataFrame({"Tstep": [500,1000,1500,3000], "Accuracy":[accd['0'], accd['1'], accd['2'], accd['3']]})
            # plot the feature importances in bars.
            plt.figure(figsize=(40,10))
            #plt.xticks(rotation=45)
            sns.set(font_scale=2)
            sns.lineplot(x="Tstep",y= "Accuracy", data=accuracy_df)
            plt.savefig(pwd+'/figures/'+args.dataset+'_accuracy.png')
            plt.tight_layout()
            plt.show()
            # logger.info('All done.')'''

    elif(args.method=="individual"):
        df = pd.DataFrame({"dataset":[],"f_split":[],"l_feat":[],"tstep":[], "accuracy ind electrodes":[],"sd ind electrodes":[],"accuracy ind features":[],"sd ind features":[],"classifier":[], "best_params":[]})

        datasets=["bci3"]
        for i in range(len(datasets)):
            args.tstep=np.random.choice(["500","3000"]),
            args.classifier=np.random.choice(["SVM","RF"]),
            args.f_split=np.random.choice([1,2,3]),
            args.l_feat=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
            args.niter=np.random.choice([200]),
            args.gen=np.random.choice([10]),
    
            acc,sd,accf,sdf,best_params=individual(args)
            #for n in range(args.gen+1):
            df = df.append({"dataset":args.dataset,"f_split":args.f_split,"l_feat":args.l_feat,"tstep":args.tstep,"accuracy ind electrodes":acc,"sd ind electrodes":sd,"accuracy ind features":accf,"sd ind features":sdf,"classifier":args.classifier,"best_params":best_params},ignore_index=True)
            log_file_name = 'accuracy_log_'+args.dataset+'.csv'
            pwd = os.getcwd()
            log_dir = pwd+'/log_dir/'
            df.to_csv(log_dir+log_file_name, index=False)

            df.to_csv(log_file_name, index=False)

            '''accuracy_df = pd.DataFrame({"Electrodes": np.arange(1,len(acc)+1), "Accuracy":list(acc.values())})
            # plot the feature importances in bars.
            plt.figure(figsize=(20,10))
            #plt.xticks(rotation=45)
            sns.set(font_scale=3)
            sns.lineplot(x="Electrodes",y= "Accuracy", data=accuracy_df)
            plt.savefig('./figures/accuracy_individual_electrodes'+args.dataset+'.png')
            plt.show()'''

            '''accuracy_n={}
            i=0
            for capital in acc.values():
                i=i+1
                accuracy_n[str(i)]=int(capital*100)

            acc_data = list(accuracy_n.items())
            an_array = np.array(acc_data)

            import scipy.io as sio
            mat_contents = sio.loadmat('./locs/'+args.dataset[:2]+'_electrodes.mat')
            locs=mat_contents["electrodes"]
            dfm=pd.DataFrame({'x Taliarich':locs[:,0],'y Taliarich':locs[:,1], 'z Taliarich':locs[:,2], 'accuracy':an_array[:,1]})
            l=[]
            for i in dfm.values[:,3]:
                i = int(i)
                l.append(i)

            plt.style.use('seaborn')
            plt.figure(figsize=(15,5))
            plt.subplot(1, 3, 1)
            plt.scatter(dfm["x Taliarich"], dfm["y Taliarich"], c=l, cmap="viridis")
            #plt.clim(vmin = 0.4, vmax = 0.45)
            plt.title('Electrode locations')
            plt.xlabel('x Taliarich')
            plt.ylabel('y Taliarich')
            clb=plt.colorbar()
            clb.set_label('Accuracy')

            plt.subplot(1, 3, 2)
            plt.scatter(dfm["y Taliarich"], dfm["z Taliarich"], c=l, cmap="viridis")
            #plt.clim(vmin = 0.4, vmax = 0.45)
            plt.title('Electrode locations')
            plt.xlabel('y Taliarich')
            plt.ylabel('z Taliarich')
            clb=plt.colorbar()
            clb.set_label('Accuracy')

            plt.subplot(1, 3, 3)
            plt.scatter(dfm["z Taliarich"], dfm["x Taliarich"], c=l, cmap="viridis")
            #plt.clim(vmin = 0.4, vmax = 0.45)
            plt.title('Electrode locations')
            plt.xlabel('z Taliarich')
            plt.ylabel('x Taliarich')
            clb=plt.colorbar()
            clb.set_label('Accuracy')

            plt.tight_layout()
            plt.savefig("./figures_electrode_locs/"+args.dataset)


            accuracy_df = pd.DataFrame({"Electrodes": np.arange(1,len(accf)+1), "Accuracy":list(accf.values())})
            # plot the feature importances in bars.
            plt.figure(figsize=(20,10))
            #plt.xticks(rotation=45)
            sns.set(font_scale=3)
            sns.lineplot(x="Electrodes",y= "Accuracy", data=accuracy_df)
            plt.savefig('./figures/accuracy_individual_features'+args.dataset+'.png')
            plt.show()'''

            '''accuracy_df = pd.DataFrame({"Tstep": [500,1000,1500,3000], "Accuracy":[accd['0'], accd['1'], accd['2'], accd['3']]})
            # plot the feature importances in bars.
            plt.figure(figsize=(40,10))
            #plt.xticks(rotation=45)
            sns.set(font_scale=2)
            sns.lineplot(x="Tstep",y= "Accuracy", data=accuracy_df)
            plt.savefig(pwd+'/figures/'+args.dataset+'_accuracy.png')
            plt.tight_layout()
            plt.show()
            # logger.info('All done.')'''

    elif(args.method=="baseline"):
        df = pd.DataFrame({"dataset":[],"l_feat":[],"tstep":[],"accuracy":[],"std":[],"f_split":[],"classifier":[],"best_params":[]})

        #datasets=["bci3","jc_mot","fp_im", "jc_im", "jm_im", "rr_im", "rh_im", "bp_im","wc_mot","zt_mot","fp_mot","gc_mot","hh_mot","hl_mot","jf_mot","jp_mot","rh_mot","rr_mot","ug_mot","jt_mot","jm_mot","gf_mot","bp_mot","cc_mot","ca_mot","de_mot"]
        datasets=["bci3"]
        for i in range(len(datasets)):
            print("ITERATION CHANGE")
            args.dataset=datasets[i]
            args.tstep=np.random.choice(["3000"])
            args.classifier=np.random.choice(["SVM","RF"])
            args.f_split=np.random.choice([1])
            args.l_feat=np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35])
            args.niter=np.random.choice([200])
            args.gen=np.random.choice([10])
            accd,sd,best_params=baseline(args)
            #for n in range(args.gen+1):
            df = df.append({"dataset":args.dataset,"l_feat":args.l_feat,"tstep":args.tstep,"accuracy":accd, "std":sd,"f_split":args.f_split,"classifier":args.classifier, "best_params":best_params},ignore_index=True)
            print(df)
            log_file_name = 'accuracy_log_'+str(int(args.seed))+'.csv'
            #pwd = os.getcwd()
            #log_dir = pwd+'/log_dir/'
            #df.to_csv(log_dir+log_file_name, index=False)

            df.to_csv(log_file_name, index=False)
            print("file saved")
            print(log_file_name)

            '''accuracy_df = pd.DataFrame({"Tstep": [500,1000,1500,3000], "Accuracy":[accd['0'], accd['1'], accd['2'], accd['3']]})
            # plot the feature importances in bars.
            plt.figure(figsize=(40,10))
            #plt.xticks(rotation=45)
            sns.set(font_scale=2)
            sns.lineplot(x="Tstep",y= "Accuracy", data=accuracy_df)
            plt.savefig(pwd+'/figures/'+args.dataset+'_accuracy.png')
            plt.tight_layout()
            plt.show()
            # logger.info('All done.')'''
           

    elif(args.method=="topn_elec"):
        df = pd.DataFrame({"dataset":[],"f_split":[],"l_feat":[],"tstep":[],"accuracy":[],"std":[],"classifier":[],"best_params":[]})

        parameters = dict(dataset=["bci3"],
        tstep=[500],classifier=["SVM","RF"],f_split=[2,3],l_feat=[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]]
        )

        n_generations=10
        #l_feat=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
        #args.gen=n_generations
        #args.l_feat=l_feat
        args.niter=200

        param_values = [v for v in parameters.values()]
        for args.dataset,args.tstep,args.classifier,args.f_split,args.l_feat in product(*param_values):
            acc,best_params,sd=topn_elec(args)
            #for n in range(args.gen+1):
            df = df.append({"dataset":args.dataset,"f_split":args.f_split,"l_feat":args.l_feat,"tstep":args.tstep,"accuracy":acc, "std":sd,"classifier":args.classifier, "best_params":best_params},ignore_index=True)
            log_file_name = 'accuracy_topn_log_'+str(args.dataset)+'.csv'
            pwd = os.getcwd()
            log_dir = pwd+'/log_dir/'
            df.to_csv(log_dir+log_file_name, index=False)

            df.to_csv(log_file_name, index=False)

            '''accuracy_df = pd.DataFrame({"Tstep": [500,1000,1500,3000], "Accuracy":[accd['0'], accd['1'], accd['2'], accd['3']]})
            # plot the feature importances in bars.
            plt.figure(figsize=(40,10))
            #plt.xticks(rotation=45)
            sns.set(font_scale=2)
            sns.lineplot(x="Tstep",y= "Accuracy", data=accuracy_df)
            plt.savefig(pwd+'/figures/'+args.dataset+'_accuracy.png')
            plt.tight_layout()
            plt.show()
            # logger.info('All done.')'''
            
    elif(args.method=="topn_feat"):
        parameters = dict(dataset=["bci3", "jc_mot"],
        tstep=[3000]
        )
        df = pd.DataFrame({"dataset":[],"f_split":[],"l_feat":[],"tstep":[],"llim":[],"nfeatures":[], "electrodes":[]})
        param_values = [v for v in parameters.values()]
        for args.dataset,args.tstep in product(*param_values):
            l_feat=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
            #args.gen=n_generations
            args.l_feat=l_feat
            llim, nfeatures, elec = topn_feat(args)
            df = df.append({"dataset":args.dataset,"f_split":args.f_split,"l_feat":args.l_feat,"tstep":args.tstep,"llim":llim, "nfeatures":nfeatures, "electrodes":elec},ignore_index=True)
            log_file_name = 'accuracy_topn_log_'+str(args.dataset)+'.csv'
            pwd = os.getcwd()
            log_dir = pwd+'/log_dir/'
            df.to_csv(log_dir+log_file_name, index=False)

            df.to_csv(log_file_name, index=False)
