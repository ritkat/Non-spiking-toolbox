# Author: Nikhil Garg
# Organization: 3IT & NECOTIS,
# Université de Sherbrooke, Canada

import itertools
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import gc
import matplotlib.pyplot as plt
import time
from utilis import *
from args import args as my_args
from encode import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics as metrics
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
# import autosklearn.classification


def evaluate_encoder(args):

    nbInputs = 64
    seed = 500
    random.seed(seed)
    np.random.seed(seed)
    print(args.__dict__)

    spike_times_train_up_list, spike_times_train_dn_list, spike_times_test_up_list, spike_times_test_dn_list, X_Train_list,X_Test_list, Y_Train_list,Y_Test_list,avg_spike_rate_list = encode(args)
    svm_score_input_list,rf_score_input_list, svm_score_baseline_list, svm_score_comb_list, rf_score_comb_list, acc_list, sel_list, gen_list, nfeat_list, rf_score_individual_input_list=[],[],[],[],[],[],[],[],[],[]
    if args.dataset=="bci3":

        nbtimepoints = int(args.duration / args.tstep)

        spike_rate_array_all_input_train = np.ones((nbInputs, nbtimepoints)) * -1  # Dummy spike counts. Would be discarded in last lines
        spike_rate_array_all_input_test = np.ones((nbInputs, nbtimepoints)) * -1


        #Training
        spike_times_up = spike_times_train_up_list[0]
        spike_times_dn = spike_times_train_dn_list[0]
        labels = Y_Train_list[0]
        label_list = []

        for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):
            # print(iteration)
            times, indices = convert_data_add_format(sample_time_up, sample_time_down)
            rate_array_input = recorded_output_to_spike_rate_array(index_array=np.array(indices),
                                                            time_array=np.array(times),
                                                            duration=args.tlast, tstep=args.tstep, nbneurons=nbInputs)

            spike_rate_array_all_input_train=np.dstack((spike_rate_array_all_input_train,rate_array_input))
            label_list.append(np.array(labels[iteration]))
            gc.collect()

        spike_rate_array_all_input_train = spike_rate_array_all_input_train[:,:,1:]

        X_input_train, Y_input_train = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_input_train, label_array=label_list,
                                                        tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)
        print("Number of Train samples : ")
        print(len(X_input_train))

        
        # Testing
        spike_times_up = spike_times_test_up_list[0]
        spike_times_dn = spike_times_test_dn_list[0]
        labels = Y_Test_list[0]
        label_list = []
        #TODO: Vectorize and predetermine the dimension of all arrays to allocate memory at start
        for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):
            #TODO: do a TQDM progress bar here
            # print(iteration)
            times, indices = convert_data_add_format(sample_time_up, sample_time_down)

            rate_array_input = recorded_output_to_spike_rate_array(index_array=np.array(indices),
                                                                time_array=np.array(times),
                                                                duration=3000, tstep=args.tstep, nbneurons=nbInputs)

            spike_rate_array_all_input_test = np.dstack((spike_rate_array_all_input_test, rate_array_input))
            label_list.append(np.array(labels[iteration]))
            gc.collect()

        spike_rate_array_all_input_test=spike_rate_array_all_input_test[:,:,1:]
        
        X_input_test, Y_input_test = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_input_test, label_array=label_list,
                                                        tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)
        
        print("Number of Test samples : ")
        print(len(X_input_test))


        X_Train_segmented, Y_Train_segmented = segment(X_Train_list[0], Y_Train_list[0],tstep= args.tstep, tstart=0, tstop=args.tlast)
        print(len(X_Train_segmented))
        X_Train_segmented=np.array(X_Train_segmented)
        Y_Train_segmented=np.array(Y_Train_segmented)
        X_Test_segmented, Y_Test_segmented = segment(X_Test_list[0], Y_Test_list[0], tstep=args.tstep, tstart=0, tstop=args.tlast)
        X_Test_segmented=np.array(X_Test_segmented)
        Y_Test_segmented=np.array(Y_Test_segmented)
        X_train = np.mean(X_Train_segmented, axis=1)
        X_test = np.mean(X_Test_segmented, axis=1)
        Y_train = Y_Train_segmented
        Y_test = Y_Test_segmented


        '''
        Input model for evaluating spike rates features obtained from temporal difference encoding
        '''
        n_iter=args.niter
        svma=svm.SVC()
        distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
        clf = RandomizedSearchCV(svma, distributions, random_state=0)
        clf.fit(X_input_train, Y_input_train)
        prediction = clf.predict(X_input_test)
        svm_score_input=metrics.accuracy_score(Y_input_test,prediction)
        print("ONEODNE")


        print("Input test accuraccy")
        print(svm_score_input)

        rf = RandomForestClassifier()
        distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
        clf = RandomizedSearchCV(rf, distributions, random_state=0, n_jobs=-1, n_iter=n_iter)
        #print("THESHAPE OF X_input_train"+len(X_input_train))
        clf.fit(X_input_train, Y_input_train)
        prediction = clf.predict(X_input_test)
        rf_score_input=metrics.accuracy_score(Y_input_test,prediction)

        gen={}
        sel=[]
        for k in range(args.gen+1):
            gen[str(k)]=[]
        estimator = RandomForestClassifier()
        selector = GeneticSelectionCV(
        estimator,
        cv=5,
        verbose=1,
        scoring="accuracy",
        max_features=5,
        n_population=300,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=args.gen,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=10,
        caching=True,
        n_jobs=100,)
        selector = selector.fit(X_input_train, Y_input_train)
        acc=selector.score(X_input_test, Y_input_test)
        tempo=np.where(selector.support_.astype(int)==1)[0]
        sel=tempo
        for k in range(args.gen+1):
            gen[str(k)].append(selector.generation_scores_[k])
        nfeat=sel.shape[0]

        print("THIS")
        X_input_train_n = np.array(X_input_train)
        for i in range(0,len(X_input_train)):
            if i==0:
                X_input_train_n=X_input_train[0]
            else:
                X_input_train_n=np.vstack((X_input_train_n, X_input_train[i]))
        print(X_input_train_n.shape)
        
        X_input_test_n = np.array(X_input_test)
        for i in range(0,len(X_input_test)):
            if i==0:
                X_input_test_n=X_input_test[0]
            else:
                X_input_test_n=np.vstack((X_input_test_n, X_input_test[i]))
    

        X_input_train_comb=np.hstack((X_input_train_n, X_train))
        X_input_test_comb=np.hstack((X_input_test_n, X_test))


        svma=svm.SVC()
        distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
        clf = RandomizedSearchCV(svma, distributions, random_state=0)
        clf.fit(X_input_train_comb, Y_input_train)
        prediction = clf.predict(X_input_test_comb)
        svm_score_comb=metrics.accuracy_score(Y_input_test,prediction)

        rf = RandomForestClassifier()
        distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
        clf = RandomizedSearchCV(rf, distributions, random_state=0, n_jobs=-1, n_iter=n_iter)
        clf.fit(X_input_train_comb, Y_input_train)
        #print("THESHAPE OF X_input_train"+len(X_input_train))
        prediction = clf.predict(X_input_test_comb)
        rf_score_comb=metrics.accuracy_score(Y_input_test,prediction)

        rf_score_individual_input=[]
        for i in range(X_input_train_n.shape[1]):
            X_temp_input=X_input_train_n[:,i]
            X_temp_input=np.reshape(X_temp_input, (X_input_train_n.shape[0],1))
            X_temp_input_test=X_input_test_n[:,i]
            X_temp_input_test=np.reshape(X_temp_input_test, (X_input_test_n.shape[0],1))
            rf = RandomForestClassifier()
            distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
            clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=100,n_jobs=-1)
            clf.fit(X_temp_input, Y_input_train)
            #print("THESHAPE OF X_input_train"+len(X_input_train))
            y_pred_rf_w = clf.predict(X_temp_input_test)
            rf_score_individual_input.append(metrics.accuracy_score(Y_input_test,y_pred_rf_w))


        
        '''cls_auto = autosklearn.classification.AutoSklearnClassifier()
        cls_auto.fit(X_input_train_comb, Y_input_train)
        predictions = cls_auto.predict(X_input_test_comb)
        from sklearn import metrics
        auto_score=metrics.accuracy_score(Y_input_test, predictions)'''


        '''pwd = os.getcwd()
        plot_dir = pwd + '/plots/'
        plt.rcParams.update({'font.size': 16})
        #Confusion matrix
        predictions = clf_input.predict(X_input_test)
        ax = skplt.metrics.plot_confusion_matrix(Y_input_test, predictions, normalize=True)
        plt.savefig(plot_dir+args.experiment_name+'_decoded_'+'confusion'+'.svg')
        plt.clf()
        #ROC curve
        predicted_probas = clf_input.predict_proba(X_input_test)
        ax2 = skplt.metrics.plot_roc(Y_input_test, predicted_probas)
        plt.savefig(plot_dir+args.experiment_name+'_decoded_'+'roc'+'.svg')
        plt.clf()'''


        '''
        Baseline model for evaluating time domain averaged features obtained from raw signals
        '''
        clf_baseline = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
        clf_baseline.fit(X_train, Y_train)
        #print("THESHAPE OF X_train"+str(X_train.shape))
        svm_score_baseline = clf_baseline.score(X_test, Y_test)
        print("Baseline accuraccy")
        print(svm_score_baseline)
        



        


        #Confusion matrix
        '''predictions = clf_baseline.predict(X_test)
        ax = skplt.metrics.plot_confusion_matrix(Y_test, predictions, normalize=True)
        plt.savefig(plot_dir+args.experiment_name+'_baseline_'+'confusion'+'.svg')
        plt.clf()
        #ROC curve
        predicted_probas = clf_baseline.predict_proba(X_test)
        ax2 = skplt.metrics.plot_roc(Y_test, predicted_probas)
        plt.savefig(plot_dir+args.experiment_name+'_baseline_'+'roc'+'.svg')
        plt.clf()'''
        
        

        #plot_dataset(X=X_test,y=Y_test,fig_dir=plot_dir, fig_name=args.experiment_name+'_dataset_raw_test',args=args)
        #plot_dataset(X=np.array(X_input_test),y=np.array(Y_input_test),fig_dir=plot_dir, fig_name=args.experiment_name+'_dataset_decoded_test',args=args)
        np.savez_compressed(
            'spike_data.npz',
            X=np.array(X_input_test),
            Y_Train=np.array(Y_input_test)
        )


        return svm_score_input,rf_score_input,avg_spike_rate, svm_score_baseline, svm_score_comb, rf_score_comb, acc, sel, gen, nfeat, rf_score_individual_input# auto_score

    else:
        #spike_times_train_up, spike_times_train_dn, spike_times_test_up, spike_times_test_dn, X_Train,X_Test, Y_Train,Y_Test,avg_spike_rate = encode(args)
        for k in range(args.kfold):
            nbtimepoints = int(args.duration / args.tstep)

            spike_rate_array_all_input_train = np.ones((nbInputs, nbtimepoints)) * -1  # Dummy spike counts. Would be discarded in last lines
            spike_rate_array_all_input_test = np.ones((nbInputs, nbtimepoints)) * -1


            #Training
            spike_times_up = spike_times_train_up_list[k]
            spike_times_dn = spike_times_train_dn_list[k]
            labels = Y_Train_list[k]
            label_list = []

            for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):
                # print(iteration)
                times, indices = convert_data_add_format(sample_time_up, sample_time_down)
                rate_array_input = recorded_output_to_spike_rate_array(index_array=np.array(indices),
                                                                time_array=np.array(times),
                                                                duration=args.tlast, tstep=args.tstep, nbneurons=nbInputs)

                spike_rate_array_all_input_train=np.dstack((spike_rate_array_all_input_train,rate_array_input))
                label_list.append(np.array(labels[iteration]))
                gc.collect()

            spike_rate_array_all_input_train = spike_rate_array_all_input_train[:,:,1:]

            X_input_train, Y_input_train = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_input_train, label_array=label_list,
                                                            tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)
            print("Number of Train samples : ")
            print(len(X_input_train))

            
            # Testing
            spike_times_up = spike_times_test_up_list[k]
            spike_times_dn = spike_times_test_dn_list[k]
            labels = Y_Test_list[k]
            label_list = []
            #TODO: Vectorize and predetermine the dimension of all arrays to allocate memory at start
            for iteration, (sample_time_up, sample_time_down) in enumerate(zip(spike_times_up, spike_times_dn)):
                #TODO: do a TQDM progress bar here
                # print(iteration)
                times, indices = convert_data_add_format(sample_time_up, sample_time_down)

                rate_array_input = recorded_output_to_spike_rate_array(index_array=np.array(indices),
                                                                    time_array=np.array(times),
                                                                    duration=3000, tstep=args.tstep, nbneurons=nbInputs)

                spike_rate_array_all_input_test = np.dstack((spike_rate_array_all_input_test, rate_array_input))
                label_list.append(np.array(labels[iteration]))
                gc.collect()

            spike_rate_array_all_input_test=spike_rate_array_all_input_test[:,:,1:]
            
            X_input_test, Y_input_test = spike_rate_array_to_features(spike_rate_array=spike_rate_array_all_input_test, label_array=label_list,
                                                            tstep=args.tstep, tstart=args.tstart, tlast=args.tlast)
            
            print("Number of Test samples : ")
            print(len(X_input_test))


            X_Train_segmented, Y_Train_segmented = segment(X_Train_list[k], Y_Train_list[k],tstep= args.tstep, tstart=0, tstop=args.tlast)
            print(len(X_Train_segmented))
            X_Train_segmented=np.array(X_Train_segmented)
            Y_Train_segmented=np.array(Y_Train_segmented)
            X_Test_segmented, Y_Test_segmented = segment(X_Test_list[k], Y_Test_list[k], tstep=args.tstep, tstart=0, tstop=args.tlast)
            X_Test_segmented=np.array(X_Test_segmented)
            Y_Test_segmented=np.array(Y_Test_segmented)
            X_train = np.mean(X_Train_segmented, axis=1)
            X_test = np.mean(X_Test_segmented, axis=1)
            Y_train = Y_Train_segmented
            Y_test = Y_Test_segmented


            '''
            Input model for evaluating spike rates features obtained from temporal difference encoding
            '''
            n_iter=args.niter
            svma=svm.SVC()
            distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
            clf = RandomizedSearchCV(svma, distributions, random_state=0)
            clf.fit(X_input_train, Y_input_train)
            prediction = clf.predict(X_input_test)
            svm_score_input=metrics.accuracy_score(Y_input_test,prediction)
            svm_score_input_list.append(svm_score_input)
            print("ONEODNE")


            print("Input test accuraccy")
            print(svm_score_input)

            rf = RandomForestClassifier()
            distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
            clf = RandomizedSearchCV(rf, distributions, random_state=0, n_jobs=-1, n_iter=n_iter)
            #print("THESHAPE OF X_input_train"+len(X_input_train))
            clf.fit(X_input_train, Y_input_train)
            prediction = clf.predict(X_input_test)
            rf_score_input=metrics.accuracy_score(Y_input_test,prediction)
            rf_score_input_list.append(rf_score_input)

            gen={}
            sel=[]
            for k in range(args.gen+1):
                gen[str(k)]=[]
            estimator = RandomForestClassifier()
            selector = GeneticSelectionCV(
            estimator,
            cv=5,
            verbose=1,
            scoring="accuracy",
            max_features=5,
            n_population=300,
            crossover_proba=0.5,
            mutation_proba=0.2,
            n_generations=args.gen,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.05,
            tournament_size=3,
            n_gen_no_change=10,
            caching=True,
            n_jobs=100,)
            selector = selector.fit(X_input_train, Y_input_train)
            acc=selector.score(X_input_test, Y_input_test)
            acc_list.append(acc)
            tempo=np.where(selector.support_.astype(int)==1)[0]
            sel=tempo
            sel_list.append(sel)
            for k in range(args.gen+1):
                gen[str(k)].append(selector.generation_scores_[k])
            gen_list.append(gen)
            nfeat=sel.shape[0]
            nfeat_list.append(nfeat)

            print("THIS")
            X_input_train_n = np.array(X_input_train)
            for i in range(0,len(X_input_train)):
                if i==0:
                    X_input_train_n=X_input_train[0]
                else:
                    X_input_train_n=np.vstack((X_input_train_n, X_input_train[i]))
            print(X_input_train_n.shape)
            
            X_input_test_n = np.array(X_input_test)
            for i in range(0,len(X_input_test)):
                if i==0:
                    X_input_test_n=X_input_test[0]
                else:
                    X_input_test_n=np.vstack((X_input_test_n, X_input_test[i]))
        

            X_input_train_comb=np.hstack((X_input_train_n, X_train))
            X_input_test_comb=np.hstack((X_input_test_n, X_test))


            svma=svm.SVC()
            distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
            clf = RandomizedSearchCV(svma, distributions, random_state=0)
            clf.fit(X_input_train_comb, Y_input_train)
            prediction = clf.predict(X_input_test_comb)
            svm_score_comb=metrics.accuracy_score(Y_input_test,prediction)
            svm_score_comb_list.append(svm_score_comb)

            rf = RandomForestClassifier()
            distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
            clf = RandomizedSearchCV(rf, distributions, random_state=0, n_jobs=-1, n_iter=n_iter)
            clf.fit(X_input_train_comb, Y_input_train)
            #print("THESHAPE OF X_input_train"+len(X_input_train))
            prediction = clf.predict(X_input_test_comb)
            rf_score_comb=metrics.accuracy_score(Y_input_test,prediction)
            rf_score_comb_list.append(rf_score_comb)

            rf_score_individual_input=[]
            for i in range(X_input_train_n.shape[1]):
                X_temp_input=X_input_train_n[:,i]
                X_temp_input=np.reshape(X_temp_input, (X_input_train_n.shape[0],1))
                X_temp_input_test=X_input_test_n[:,i]
                X_temp_input_test=np.reshape(X_temp_input_test, (X_input_test_n.shape[0],1))
                rf = RandomForestClassifier()
                distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
                clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=100,n_jobs=-1)
                clf.fit(X_temp_input, Y_input_train)
                #print("THESHAPE OF X_input_train"+len(X_input_train))
                y_pred_rf_w = clf.predict(X_temp_input_test)
                rf_score_individual_input.append(metrics.accuracy_score(Y_input_test,y_pred_rf_w))
            rf_score_individual_input_list.append(rf_score_individual_input)

            
            '''cls_auto = autosklearn.classification.AutoSklearnClassifier()
            cls_auto.fit(X_input_train_comb, Y_input_train)
            predictions = cls_auto.predict(X_input_test_comb)
            from sklearn import metrics
            auto_score=metrics.accuracy_score(Y_input_test, predictions)'''


            '''pwd = os.getcwd()
            plot_dir = pwd + '/plots/'
            plt.rcParams.update({'font.size': 16})
            #Confusion matrix
            predictions = clf_input.predict(X_input_test)
            ax = skplt.metrics.plot_confusion_matrix(Y_input_test, predictions, normalize=True)
            plt.savefig(plot_dir+args.experiment_name+'_decoded_'+'confusion'+'.svg')
            plt.clf()
            #ROC curve
            predicted_probas = clf_input.predict_proba(X_input_test)
            ax2 = skplt.metrics.plot_roc(Y_input_test, predicted_probas)
            plt.savefig(plot_dir+args.experiment_name+'_decoded_'+'roc'+'.svg')
            plt.clf()'''


            '''
            Baseline model for evaluating time domain averaged features obtained from raw signals
            '''
            clf_baseline = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
            clf_baseline.fit(X_train, Y_train)
            #print("THESHAPE OF X_train"+str(X_train.shape))
            svm_score_baseline = clf_baseline.score(X_test, Y_test)
            svm_score_baseline_list.append(svm_score_baseline)
            print("Baseline accuraccy")
            print(svm_score_baseline)
            



            


            #Confusion matrix
            '''predictions = clf_baseline.predict(X_test)
            ax = skplt.metrics.plot_confusion_matrix(Y_test, predictions, normalize=True)
            plt.savefig(plot_dir+args.experiment_name+'_baseline_'+'confusion'+'.svg')
            plt.clf()
            #ROC curve
            predicted_probas = clf_baseline.predict_proba(X_test)
            ax2 = skplt.metrics.plot_roc(Y_test, predicted_probas)
            plt.savefig(plot_dir+args.experiment_name+'_baseline_'+'roc'+'.svg')
            plt.clf()'''
            
            

            #plot_dataset(X=X_test,y=Y_test,fig_dir=plot_dir, fig_name=args.experiment_name+'_dataset_raw_test',args=args)
            #plot_dataset(X=np.array(X_input_test),y=np.array(Y_input_test),fig_dir=plot_dir, fig_name=args.experiment_name+'_dataset_decoded_test',args=args)
            np.savez_compressed(
                'spike_data.npz',
                X=np.array(X_input_test),
                Y_Train=np.array(Y_input_test)
            )


        return svm_score_input_list,rf_score_input_list,avg_spike_rate_list, svm_score_baseline_list, svm_score_comb_list, rf_score_comb_list, acc_list, sel_list, gen_list, nfeat_list, rf_score_individual_input_list# auto_score

        
if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    args = my_args()
    print(args.__dict__)
    logging.basicConfig(level=logging.DEBUG)
    # Fix the seed of all random number generator
    seed = 500
    random.seed(seed)
    np.random.seed(seed)
    svm_score_input,avg_spike_rate, svm_score_baseline= evaluate_encoder(args)
    print('Average spike rate :')
    print(avg_spike_rate)
    print('Accuraccy input: ' + str(svm_score_input))
    logger.info('All done.')
