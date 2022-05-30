from utils import*
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
from genetic_selection import GeneticSelectionCV
import seaborn as sns
import os
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
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from createFV import *

# Import the RFE from sklearn library
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from genetic_selection import GeneticSelectionCV
import seaborn as sns
import os
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from args_final import args as my_args

args=my_args()

def genetic(args):
  if(args.dataset=="bci3"):
    data=np.load('./data/bci_3.npz')
    data_train=data["X"]
    data_test=data["X_test"]
    labels=data['events']
    truelabels=np.loadtxt("./true_labels.txt", delimiter="/n")
    data_train_ib_500=segment(data_train, segment_length=500)
    segment_length=500
    labels_train_ib_500=np.repeat(labels,3000/int(segment_length))

    data_test_ib_500=segment(data_test, segment_length=500)
    segment_length=500
    labels_test_ib_500=np.repeat(truelabels,3000/int(segment_length))

    #1000 Tstep
    data_train_ib_1000=segment(data_train, segment_length=1000)
    segment_length=1000
    labels_train_ib_1000=np.repeat(labels,3000/int(segment_length))

    data_test_ib_1000=segment(data_test, segment_length=1000)
    segment_length=1000
    labels_test_ib_1000=np.repeat(truelabels,3000/int(segment_length))

    #1500 Tstep
    data_train_ib_1500=segment(data_train, segment_length=1500)
    segment_length=1500
    labels_train_ib_1500=np.repeat(labels,3000/int(segment_length))

    data_test_ib_1500=segment(data_test, segment_length=1500)
    segment_length=1500
    labels_test_ib_1500=np.repeat(truelabels,3000/int(segment_length))

    data_train_ib_3000=data_train
    segment_length=3000
    labels_train_ib_3000=labels

    data_test_ib_3000=data_test
    segment_length=3000
    labels_test_ib_3000=truelabels

    training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
    label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
    testing_data={'500':data_test_ib_500, '1000':data_test_ib_1000, '1500':data_test_ib_1500, '3000':data_test_ib_3000}
    label_data_test={'500':labels_test_ib_500, '1000':labels_test_ib_1000, '1500':labels_test_ib_1500, '3000':labels_test_ib_3000}
    segment_length=[500,1000,1500,3000]
    l_feat=args.l_feat 

    data_train_loop=training_data[str(args.tstep)]
    labels_train_loop=label_data[str(args.tstep)]
    data_test_loop=testing_data[str(args.tstep)]
    labels_test_loop=label_data_test[str(args.tstep)]

    acc=[]
    l_feat=args.l_feat

    gen={}
    genstd={}
    sel=[]
    #n_generations=40
    for k in range(args.gen+1):
      gen[str(k)]=[]  
    for k in range(args.gen+1):
      genstd[str(k)]=[]  
    f_split=args.f_split
    df_train_temp, df_test_temp=createFV_individual(data_train_loop, data_test_loop,f_split, 1000, l_feat, True)
    print(np.amax(df_train_temp.values))
    print(np.amin(df_train_temp.values))
    # Without feature selection check accuracy with Random forest
    if args.classifier=="RF":
      estimator = RandomForestClassifier()
    else:
      object=StandardScaler()
      object.fit(df_train_temp)
      df_train_temp = object.transform(df_train_temp) 
      df_test_temp=object.transform(df_test_temp) 
      estimator=svm.SVC()
    selector = GeneticSelectionCV(
    estimator,
    cv=5,
    verbose=1,
    scoring="accuracy",
    max_features=args.max_feat,
    n_population=300,
    crossover_proba=0.5,
    mutation_proba=0.2,
    n_generations=args.gen,
    crossover_independent_proba=0.5,
    mutation_independent_proba=0.05,
    tournament_size=3,
    n_gen_no_change=None,
    caching=True,
    n_jobs=-1,)
    selector = selector.fit(df_train_temp.values, labels_train_loop)
    params=selector.estimator_
    tempo=np.where(selector.support_.astype(int)==1)[0]
    sel.append(tempo)

    acc.append(selector.score(df_test_temp.values, labels_test_loop))
    for k in range(args.gen+1):
      gen[str(k)].append(selector.generation_scores_[k])

    self=sel
    nfeat=len(self[0])
    accd=acc
    sd=0
    for k in range(args.gen+1):
      gen[str(k)]=sum(gen[str(k)])/len(gen[str(k)])
    for k in range(args.gen+1):
      genstd[str(k)]=0

    return accd, gen, self, nfeat, sd, genstd, params

  elif(args.dataset=="speech"):
    data_ib=np.load('./data/speech.npz')
    data_train_ib = data_ib["X_Train"]
    labels_train_ib = data_ib["Y_Train"]
    #500 Tstep
    data_train_ib_500, rep=segment_speech(data_train_ib, segment_length=500)
    print(np.amax(data_train_ib_500))
    print(np.amin(data_train_ib_500))

    segment_length=500
    labels_train_ib_500=repeater(labels_train_ib, rep)

    #1000 Tstep
    data_train_ib_1000, rep=segment_speech(data_train_ib, segment_length=1000)
    segment_length=1000
    labels_train_ib_1000=repeater(labels_train_ib, rep)

    #1500 Tstep
    '''data_train_ib_1500=segment_speech(data_train_ib, segment_length=1500)
    segment_length=1500
    labels_train_ib_1500=repeater(labels_train_ib, rep)

    #3000 Tstep
    data_train_ib_3000=data_train_ib
    segment_length=3000
    labels_train_ib_3000=labels_train_ib'''

    training_data={'500':data_train_ib_500, '1000':data_train_ib_1000}
    label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000}
    segment_length=[500,1000]

    kf3 = KFold(n_splits=5, shuffle=False)

    #print("iteration "+str(i))
    data_train_loop=training_data[str(args.tstep)]
    labels_train_loop=label_data[str(args.tstep)]
    #fs=int(segment_length[i]/3)
    acc=[]
    l_feat=args.l_feat
    gen={}
    genstd={}
    sel=[]
    #n_generations=40
    for k in range(args.gen+1):
      gen[str(k)]=[]  
    for k in range(args.gen+1):
      genstd[str(k)]=[]  
    f_split=args.f_split
    for train_index, test_index in kf3.split(data_train_loop):
        df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index], data_train_loop[test_index], f_split,3052, l_feat, True)
        print(np.amax(df_train_temp.values))
        print(np.amin(df_train_temp.values))
        # Without feature selection check accuracy with Random forest
        if args.classifier=="RF":        
          estimator = RandomForestClassifier(n_estimators=800)
        else:
          object=StandardScaler()
          object.fit(df_train_temp)
          df_train_temp = object.transform(df_train_temp) 
          df_test_temp=object.transform(df_test_temp) 
          estimator=svm.SVC(gamma="auto")
        selector = GeneticSelectionCV(
        estimator,
        cv=5,
        verbose=1,
        scoring="accuracy",
        max_features=args.max_feat,
        n_population=300,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=args.gen,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=None,
        caching=True,
        n_jobs=-1,)
        selector = selector.fit(df_train_temp.values, labels_train_loop[train_index])
        params=selector.estimator_
        tempo=np.where(selector.support_.astype(int)==1)[0]
        sel.append(tempo)

        acc.append(selector.score(df_test_temp.values, labels_train_loop[test_index]))
        for k in range(args.gen+1):
          gen[str(k)].append(selector.generation_scores_[k])
    
    self=sel[1]
    nfeat=self.shape[0]
    accd=sum(acc)/len(acc)
    sd=np.std(acc)
    for k in range(args.gen+1):
      genstd[str(k)]=np.std(gen[str(k)])
    for k in range(args.gen+1):
      gen[str(k)]=sum(gen[str(k)])/len(gen[str(k)])


    

    return accd, gen, self, nfeat, sd, genstd, params


  else:

    data_ib=np.load('./data/'+args.dataset+'_epochs.npz')
    data_train_ib = data_ib["X"]
    labels_train_ib = data_ib["y"]
    #500 Tstep
    data_train_ib_500=segment(data_train_ib, segment_length=500)
    print(np.amax(data_train_ib_500))
    print(np.amin(data_train_ib_500))

    segment_length=500
    labels_train_ib_500=np.repeat(labels_train_ib,3000/int(segment_length))

    #1000 Tstep
    data_train_ib_1000=segment(data_train_ib, segment_length=1000)
    segment_length=1000
    labels_train_ib_1000=np.repeat(labels_train_ib,3000/int(segment_length))

    #1500 Tstep
    data_train_ib_1500=segment(data_train_ib, segment_length=1500)
    segment_length=1500
    labels_train_ib_1500=np.repeat(labels_train_ib,3000/int(segment_length))

    #3000 Tstep
    data_train_ib_3000=data_train_ib
    segment_length=3000
    labels_train_ib_3000=labels_train_ib

    training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
    label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
    segment_length=[500,1000,1500,3000]


    kf3 = KFold(n_splits=5, shuffle=False)
    #accd={}

    #print("iteration "+str(i))
    data_train_loop=training_data[str(args.tstep)]
    labels_train_loop=label_data[str(args.tstep)]
    #fs=int(segment_length[i]/3)
    acc=[]
    l_feat=args.l_feat
    gen={}
    genstd={}
    sel=[]
    #n_generations=40
    for k in range(args.gen+1):
      gen[str(k)]=[]  
    for k in range(args.gen+1):
      genstd[str(k)]=[]  
    f_split=args.f_split
    for train_index, test_index in kf3.split(data_train_loop):
        df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index], data_train_loop[test_index], f_split,1000, l_feat, True)
        print(np.amax(df_train_temp.values))
        print(np.amin(df_train_temp.values))
        # Without feature selection check accuracy with Random forest
        if args.classifier=="RF":        
          estimator = RandomForestClassifier(n_estimators=800)
        else:
          object=StandardScaler()
          object.fit(df_train_temp)
          df_train_temp = object.transform(df_train_temp) 
          df_test_temp=object.transform(df_test_temp) 
          estimator=svm.SVC(gamma="auto")
        selector = GeneticSelectionCV(
        estimator,
        cv=5,
        verbose=1,
        scoring="accuracy",
        max_features=args.max_feat,
        n_population=300,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=args.gen,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05,
        tournament_size=3,
        n_gen_no_change=None,
        caching=True,
        n_jobs=-1,)
        selector = selector.fit(df_train_temp.values, labels_train_loop[train_index])
        params=selector.estimator_
        tempo=np.where(selector.support_.astype(int)==1)[0]
        sel.append(tempo)

        acc.append(selector.score(df_test_temp.values, labels_train_loop[test_index]))
        for k in range(args.gen+1):
          gen[str(k)].append(selector.generation_scores_[k])
    
    self=sel[1]
    nfeat=self.shape[0]
    accd=sum(acc)/len(acc)
    sd=np.std(acc)
    for k in range(args.gen+1):
      genstd[str(k)]=np.std(gen[str(k)])
    for k in range(args.gen+1):
      gen[str(k)]=sum(gen[str(k)])/len(gen[str(k)])


    

    return accd, gen, self, nfeat, sd, genstd, params

def baseline(args):
  if args.dataset=="bci3":
    data=np.load('./data/bci_3.npz')
    data_train=data["X"]
    data_test=data["X_test"]
    labels=data['events']
    truelabels=np.loadtxt("./true_labels.txt", delimiter="/n")
    data_train_ib_500=segment(data_train, segment_length=500)
    segment_length=500
    labels_train_ib_500=np.repeat(labels,3000/int(segment_length))

    data_test_ib_500=segment(data_test, segment_length=500)
    segment_length=500
    labels_test_ib_500=np.repeat(truelabels,3000/int(segment_length))

    #1000 Tstep
    data_train_ib_1000=segment(data_train, segment_length=1000)
    segment_length=1000
    labels_train_ib_1000=np.repeat(labels,3000/int(segment_length))

    data_test_ib_1000=segment(data_test, segment_length=1000)
    segment_length=1000
    labels_test_ib_1000=np.repeat(truelabels,3000/int(segment_length))

    #1500 Tstep
    data_train_ib_1500=segment(data_train, segment_length=1500)
    segment_length=1500
    labels_train_ib_1500=np.repeat(labels,3000/int(segment_length))

    data_test_ib_1500=segment(data_test, segment_length=1500)
    segment_length=1500
    labels_test_ib_1500=np.repeat(truelabels,3000/int(segment_length))

    data_train_ib_3000=data_train
    segment_length=3000
    labels_train_ib_3000=labels

    data_test_ib_3000=data_test
    segment_length=3000
    labels_test_ib_3000=truelabels

    training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
    label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
    testing_data={'500':data_test_ib_500, '1000':data_test_ib_1000, '1500':data_test_ib_1500, '3000':data_test_ib_3000}
    label_data_test={'500':labels_test_ib_500, '1000':labels_test_ib_1000, '1500':labels_test_ib_1500, '3000':labels_test_ib_3000}
    segment_length=[500,1000,1500,3000]
    l_feat=args.l_feat 
    n_iter=args.niter
    f_split=args.f_split

    data_train_loop=training_data[str(args.tstep)]
    labels_train_loop=label_data[str(args.tstep)]
    data_test_loop=testing_data[str(args.tstep)]
    labels_test_loop=label_data_test[str(args.tstep)]

    df_train_temp, df_test_temp=createFV_individual(data_train_loop, data_test_loop,f_split, 1000, l_feat, True)
    print(np.amax(df_train_temp.values))
    print(np.amin(df_train_temp.values))
    # Without feature selection check accuracy with Random forest
    if(args.classifier=="RF"):
        rf = RandomForestClassifier()
        distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
        clf = RandomizedSearchCV(rf, distributions, random_state=0, n_jobs=-1, n_iter=n_iter)
        clf.fit(df_train_temp.values, labels_train_loop)
        best_params=clf.best_params_
        pred = clf.predict(df_test_temp.values)
        acc=metrics.accuracy_score(labels_test_loop,pred)
        sd=0
    elif(args.classifier=="SVM"):
        object=StandardScaler()
        object.fit(df_train_temp)
        df_train_temp = object.transform(df_train_temp) 
        df_test_temp=object.transform(df_test_temp) 
        svma=svm.SVC()
        distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
        clf = RandomizedSearchCV(svma, distributions, random_state=0, n_jobs=-1, n_iter=n_iter)
        clf.fit(df_train_temp, labels_train_loop)
        best_params=clf.best_params_
        pred = clf.predict(df_test_temp)
        acc=metrics.accuracy_score(labels_test_loop,pred)
        sd=0

    return acc,sd, best_params

  elif(args.dataset=="speech"):
    data_ib=np.load('./data/speech.npz')
    data_train_ib = data_ib["X_Train"]
    labels_train_ib = data_ib["Y_Train"]
    #500 Tstep
    data_train_ib_500, rep=segment_speech(data_train_ib, segment_length=500)
    print(np.amax(data_train_ib_500))
    print(np.amin(data_train_ib_500))

    segment_length=500
    labels_train_ib_500=repeater(labels_train_ib, rep)

    #1000 Tstep
    data_train_ib_1000, rep=segment_speech(data_train_ib, segment_length=1000)
    segment_length=1000
    labels_train_ib_1000=repeater(labels_train_ib, rep)

    '''#1500 Tstep
    data_train_ib_1500=segment_speech(data_train_ib, segment_length=1500)
    segment_length=1500
    labels_train_ib_1500=repeater(labels_train_ib, rep)

    #3000 Tstep
    data_train_ib_3000=data_train_ib
    segment_length=3000
    labels_train_ib_3000=labels_train_ib'''

    training_data={'500':data_train_ib_500, '1000':data_train_ib_1000}
    label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000}
    segment_length=[500,1000]


    kf3 = StratifiedKFold(n_splits=5, shuffle=False)
    #accd={}

    #print("iteration "+str(i))
    data_train_loop=training_data[str(args.tstep)]
    labels_train_loop=label_data[str(args.tstep)]
    #fs=int(segment_length[i]/3)
    acc=[]
    l_feat=args.l_feat 
    n_iter=args.niter
    f_split=args.f_split
    for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
        df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index], data_train_loop[test_index],f_split, 3052, l_feat, True)
        print(np.amax(df_train_temp.values))
        print(np.amin(df_train_temp.values))
        # Without feature selection check accuracy with Random forest
        if(args.classifier=="RF"):
            rf = RandomForestClassifier()
            distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
            clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
            clf.fit(df_train_temp.values, labels_train_loop[train_index])
            best_params=clf.best_params_
            pred = clf.predict(df_test_temp.values)
            acc.append(metrics.accuracy_score(labels_train_loop[test_index],pred))
        elif(args.classifier=="SVM"):
            object=StandardScaler()
            object.fit(df_train_temp)
            df_train_temp = object.transform(df_train_temp) 
            df_test_temp=object.transform(df_test_temp) 
            svma=svm.SVC()
            distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
            clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
            clf.fit(df_train_temp, labels_train_loop[train_index])
            best_params=clf.best_params_
            pred = clf.predict(df_test_temp)
            acc.append(metrics.accuracy_score(labels_train_loop[test_index],pred))
    
    sd=np.std(acc)
    accd=sum(acc)/len(acc)
    

    

    return accd, sd, best_params
    
  else:
    data_ib=np.load('./data/'+args.dataset+'_epochs.npz')
    data_train_ib = data_ib["X"]
    labels_train_ib = data_ib["y"]
    #500 Tstep
    data_train_ib_500=segment(data_train_ib, segment_length=500)
    print(np.amax(data_train_ib_500))
    print(np.amin(data_train_ib_500))

    segment_length=500
    labels_train_ib_500=np.repeat(labels_train_ib,3000/int(segment_length))

    #1000 Tstep
    data_train_ib_1000=segment(data_train_ib, segment_length=1000)
    segment_length=1000
    labels_train_ib_1000=np.repeat(labels_train_ib,3000/int(segment_length))

    #1500 Tstep
    data_train_ib_1500=segment(data_train_ib, segment_length=1500)
    segment_length=1500
    labels_train_ib_1500=np.repeat(labels_train_ib,3000/int(segment_length))

    #3000 Tstep
    data_train_ib_3000=data_train_ib
    segment_length=3000
    labels_train_ib_3000=labels_train_ib

    training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
    label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
    segment_length=[500,1000,1500,3000]


    kf3 = StratifiedKFold(n_splits=5, shuffle=False)
    #accd={}

    #print("iteration "+str(i))
    data_train_loop=training_data[str(args.tstep)]
    labels_train_loop=label_data[str(args.tstep)]
    #fs=int(segment_length[i]/3)
    acc=[]
    l_feat=args.l_feat 
    n_iter=args.niter
    f_split=args.f_split
    for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
        df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index], data_train_loop[test_index], f_split,1000, l_feat, True)
        print(np.amax(df_train_temp.values))
        print(np.amin(df_train_temp.values))
        # Without feature selection check accuracy with Random forest
        if(args.classifier=="RF"):
            rf = RandomForestClassifier()
            distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
            clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
            clf.fit(df_train_temp.values, labels_train_loop[train_index])
            best_params=clf.best_params_
            pred = clf.predict(df_test_temp.values)
            acc.append(metrics.accuracy_score(labels_train_loop[test_index],pred))
        elif(args.classifier=="SVM"):
            object=StandardScaler()
            object.fit(df_train_temp)
            df_train_temp = object.transform(df_train_temp) 
            df_test_temp=object.transform(df_test_temp) 
            svma=svm.SVC()
            distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
            clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
            clf.fit(df_train_temp, labels_train_loop[train_index])
            best_params=clf.best_params_
            pred = clf.predict(df_test_temp)
            acc.append(metrics.accuracy_score(labels_train_loop[test_index],pred))
    
    sd=np.std(acc)
    accd=sum(acc)/len(acc)
    

    

    return accd, sd, best_params


def individual(args):
    if(args.dataset=="bci3"):
        data=np.load('./data/bci_3.npz')
        data_train=data["X"]
        data_test=data["X_test"]
        labels=data['events']
        truelabels=np.loadtxt("./true_labels.txt", delimiter="/n")
        data_train_ib_500=segment(data_train, segment_length=500)
        segment_length=500
        labels_train_ib_500=np.repeat(labels,3000/int(segment_length))

        data_test_ib_500=segment(data_test, segment_length=500)
        segment_length=500
        labels_test_ib_500=np.repeat(truelabels,3000/int(segment_length))

        #1000 Tstep
        data_train_ib_1000=segment(data_train, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=np.repeat(labels,3000/int(segment_length))

        data_test_ib_1000=segment(data_test, segment_length=1000)
        segment_length=1000
        labels_test_ib_1000=np.repeat(truelabels,3000/int(segment_length))

        #1500 Tstep
        data_train_ib_1500=segment(data_train, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=np.repeat(labels,3000/int(segment_length))

        data_test_ib_1500=segment(data_test, segment_length=1500)
        segment_length=1500
        labels_test_ib_1500=np.repeat(truelabels,3000/int(segment_length))

        data_train_ib_3000=data_train
        segment_length=3000
        labels_train_ib_3000=labels

        data_test_ib_3000=data_test
        segment_length=3000
        labels_test_ib_3000=truelabels

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
        testing_data={'500':data_test_ib_500, '1000':data_test_ib_1000, '1500':data_test_ib_1500, '3000':data_test_ib_3000}
        label_data_test={'500':labels_test_ib_500, '1000':labels_test_ib_1000, '1500':labels_test_ib_1500, '3000':labels_test_ib_3000}
        segment_length=[500,1000,1500,3000]
        l_feat=args.l_feat 
        n_iter=args.niter

        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]
        data_test_loop=testing_data[str(args.tstep)]
        labels_test_loop=label_data_test[str(args.tstep)]

        acc={}
        best_params_ie={}
        for k in range(data_train_loop.shape[1]):
            acc[str(k)]=[]
        for k in range(data_train_loop.shape[1]):
            best_params_ie[str(k)]=[]
            
        f_split=args.f_split
        for i in range(data_train_loop.shape[1]):
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[:,np.newaxis,i,:], data_test_loop[:,np.newaxis,i,:],f_split, 1000, l_feat, False)
                print(np.amax(df_train_temp.values))
                print(np.amin(df_train_temp.values))
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop)
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp.values)
                    acc[str(i)].append(metrics.accuracy_score(labels_test_loop,predictions))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop)
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp)
                    acc[str(i)].append(metrics.accuracy_score(labels_test_loop,predictions))
            # Without feature selection check accuracy with Random forest    
        for k in range(data_train_loop.shape[1]):
            acc[str(k)]=sum(acc[str(k)])/len(acc[str(k)]) 


        n_features=args.nfeatures
        accf={}
        for k in range(n_features):
            accf[str(k)]=[]  
        f_split=args.f_split
        for i in range(n_features):
            df_train_temp, df_test_temp=createFV_individual(data_train_loop, data_test_loop,f_split, 1000, [i], True)
            print(np.amax(df_train_temp.values))
            print(np.amin(df_train_temp.values))
            if(args.classifier=="RF"):
                rf = RandomForestClassifier()
                distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                clf.fit(df_train_temp.values, labels_train_loop)
                predictions = clf.predict(df_test_temp.values)
                accf[str(i)].append(metrics.accuracy_score(labels_train_loop,predictions))
            elif(args.classifier=="SVM"):
                object=StandardScaler()
                object.fit(df_train_temp)
                df_train_temp = object.transform(df_train_temp) 
                df_test_temp=object.transform(df_test_temp) 
                svma=svm.SVC()
                distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                clf.fit(df_train_temp, labels_train_loop)          
                predictions = clf.predict(df_test_temp)
                accf[str(i)].append(metrics.accuracy_score(labels_train_loop,predictions))
        # Without feature selection check accuracy with Random forest    
        for k in range(n_features):
            accf[str(k)]=sum(accf[str(k)])/len(accf[str(k)]) 
            
        sd=0
        sdf=0

        return acc, sd, accf, sdf, best_params_ie

    elif(args.dataset=="speech"):
        data_ib=np.load('./data/speech.npz')
        data_train_ib = data_ib["X_Train"]
        labels_train_ib = data_ib["Y_Train"]
        #500 Tstep
        data_train_ib_500, rep=segment_speech(data_train_ib, segment_length=500)
        print(np.amax(data_train_ib_500))
        print(np.amin(data_train_ib_500))

        segment_length=500
        labels_train_ib_500=repeater(labels_train_ib, rep)

        #1000 Tstep
        data_train_ib_1000, rep=segment_speech(data_train_ib, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=repeater(labels_train_ib, rep)

        '''#1500 Tstep
        data_train_ib_1500=segment_speech(data_train_ib, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=repeater(labels_train_ib, rep)

        #3000 Tstep
        data_train_ib_3000=data_train_ib
        segment_length=3000
        labels_train_ib_3000=labels_train_ib'''

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000}
        segment_length=[500,1000]


        kf3 = StratifiedKFold(n_splits=5, shuffle=False)

        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]
        #fs=int(segment_length[i]/3)
        acc={}
        sd={}
        best_params_ie={}
        l_feat=args.l_feat
        n_iter=args.niter
        #n_generations=40
        for k in range(data_train_loop.shape[1]):
            acc[str(k)]=[]
        for k in range(data_train_loop.shape[1]):
            sd[str(k)]=[] 
        for k in range(data_train_loop.shape[1]):
            best_params_ie[str(k)]=[]
        f_split=args.f_split
        for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
            for i in range(data_train_loop.shape[1]):
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index,np.newaxis,i,:], data_train_loop[test_index,np.newaxis,i,:],f_split, 1000, l_feat, False)
                print(np.amax(df_train_temp.values))
                print(np.amin(df_train_temp.values))
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop[train_index])
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp.values)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop[train_index])
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
            # Without feature selection check accuracy with Random forest   
        for k in range(data_train_loop.shape[1]):
            sd[str(k)]=np.std(acc[str(k)])
        '''for k in range(data_train_loop.shape[1]):
            acc[str(k)]=sum(acc[str(k)])/len(acc[str(k)])'''
            


        n_features=args.nfeatures
        accf={}
        
        sdf={}
        for k in range(n_features):
            accf[str(k)]=[]  
        for k in range(n_features):
            sdf[str(k)]=[]  
        f_split=args.f_split
        for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
            for i in range(n_features):
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index], data_train_loop[test_index],f_split, 1000, [i], True)
                print(np.amax(df_train_temp.values))
                print(np.amin(df_train_temp.values))
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop[train_index])                
                    predictions = clf.predict(df_test_temp.values)
                    accf[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop[train_index])                
                    predictions = clf.predict(df_test_temp)
                    accf[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
            # Without feature selection check accuracy with Random forest  
        for k in range(n_features):
            sdf[str(k)]=np.std(accf[str(k)])
        '''for k in range(n_features):
            accf[str(k)]=sum(accf[str(k)])/len(accf[str(k)])'''
        


        

        return acc, sd, accf, sdf, best_params_ie

    else:
        data_ib=np.load('./data/'+args.dataset+'_epochs.npz')
        data_train_ib = data_ib["X"]
        labels_train_ib = data_ib["y"]
        #500 Tstep
        data_train_ib_500=segment(data_train_ib, segment_length=500)
        print(np.amax(data_train_ib_500))
        print(np.amin(data_train_ib_500))

        segment_length=500
        labels_train_ib_500=np.repeat(labels_train_ib,3000/int(segment_length))

        #1000 Tstep
        data_train_ib_1000=segment(data_train_ib, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=np.repeat(labels_train_ib,3000/int(segment_length))

        #1500 Tstep
        data_train_ib_1500=segment(data_train_ib, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=np.repeat(labels_train_ib,3000/int(segment_length))

        #3000 Tstep
        data_train_ib_3000=data_train_ib
        segment_length=3000
        labels_train_ib_3000=labels_train_ib

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
        segment_length=[500,1000,1500,3000]


        kf3 = StratifiedKFold(n_splits=5, shuffle=False)
        #accd={}

        #print("iteration "+str(i))
        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]
        #fs=int(segment_length[i]/3)
        acc={}
        sd={}
        best_params_ie={}
        l_feat=args.l_feat
        n_iter=args.niter
        #n_generations=40
        for k in range(data_train_loop.shape[1]):
            acc[str(k)]=[]
        for k in range(data_train_loop.shape[1]):
            sd[str(k)]=[] 
        for k in range(data_train_loop.shape[1]):
            best_params_ie[str(k)]=[]
        f_split=args.f_split
        for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
            for i in range(data_train_loop.shape[1]):
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index,np.newaxis,i,:], data_train_loop[test_index,np.newaxis,i,:],f_split, 1000, l_feat, False)
                print(np.amax(df_train_temp.values))
                print(np.amin(df_train_temp.values))
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop[train_index])
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp.values)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop[train_index])
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
            # Without feature selection check accuracy with Random forest   
        for k in range(data_train_loop.shape[1]):
            sd[str(k)]=np.std(acc[str(k)])
        '''for k in range(data_train_loop.shape[1]):
            acc[str(k)]=sum(acc[str(k)])/len(acc[str(k)])'''
            


        n_features=args.nfeatures
        accf={}
        
        sdf={}
        for k in range(n_features):
            accf[str(k)]=[]  
        for k in range(n_features):
            sdf[str(k)]=[]  
        f_split=args.f_split
        for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
            for i in range(n_features):
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index], data_train_loop[test_index],f_split, 1000, [i], True)
                print(np.amax(df_train_temp.values))
                print(np.amin(df_train_temp.values))
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop[train_index])                
                    predictions = clf.predict(df_test_temp.values)
                    accf[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop[train_index])                
                    predictions = clf.predict(df_test_temp)
                    accf[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
            # Without feature selection check accuracy with Random forest  
        for k in range(n_features):
            sdf[str(k)]=np.std(accf[str(k)])
        '''for k in range(n_features):
            accf[str(k)]=sum(accf[str(k)])/len(accf[str(k)])'''
        


        

        return acc, sd, accf, sdf, best_params_ie
      
  
        #sb="jc_mot"
    
def topn_elec(args):
    if args.dataset=="bci3":
        data=np.load('./data/bci_3.npz')
        data_train=data["X"]
        data_test=data["X_test"]
        labels=data['events']
        truelabels=np.loadtxt("./true_labels.txt", delimiter="/n")
        data_train_ib_500=segment(data_train, segment_length=500)
        segment_length=500
        labels_train_ib_500=np.repeat(labels,3000/int(segment_length))

        data_test_ib_500=segment(data_test, segment_length=500)
        segment_length=500
        labels_test_ib_500=np.repeat(truelabels,3000/int(segment_length))

        #1000 Tstep
        data_train_ib_1000=segment(data_train, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=np.repeat(labels,3000/int(segment_length))

        data_test_ib_1000=segment(data_test, segment_length=1000)
        segment_length=1000
        labels_test_ib_1000=np.repeat(truelabels,3000/int(segment_length))

        #1500 Tstep
        data_train_ib_1500=segment(data_train, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=np.repeat(labels,3000/int(segment_length))

        data_test_ib_1500=segment(data_test, segment_length=1500)
        segment_length=1500
        labels_test_ib_1500=np.repeat(truelabels,3000/int(segment_length))

        data_train_ib_3000=data_train
        segment_length=3000
        labels_train_ib_3000=labels

        data_test_ib_3000=data_test
        segment_length=3000
        labels_test_ib_3000=truelabels

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
        testing_data={'500':data_test_ib_500, '1000':data_test_ib_1000, '1500':data_test_ib_1500, '3000':data_test_ib_3000}
        label_data_test={'500':labels_test_ib_500, '1000':labels_test_ib_1000, '1500':labels_test_ib_1500, '3000':labels_test_ib_3000}
        segment_length=[500,1000,1500,3000]
        l_feat=args.l_feat 
        f_split=args.f_split
        n_iter=args.niter

        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]
        data_test_loop=testing_data[str(args.tstep)]
        labels_test_loop=label_data_test[str(args.tstep)]
        f_split=args.f_split
        df_train_temp, df_test_temp=createFV_individual(data_train_loop, data_test_loop,f_split, 1000, l_feat, True)
        print(np.amax(df_train_temp.values))
        print(np.amin(df_train_temp.values))
        # Without feature selection check accuracy with Random forest
        rf = RandomForestClassifier()
        '''distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
        clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)'''
        df_train, llim, nfeatures =createFV_individual_feat(data_train_loop,f_split, 1000, l_feat, True)
        rf.fit(df_train.values, labels_train_loop)
        
        # Without feature selection check auuracy with Random forest

        importances = rf.feature_importances_

        #importance per electrode
        n_electrodes=data_train_loop.shape[1]
        #dictionary of features
        #nfeatures=[]
        f1=nfeatures[0]/n_electrodes
        f2=nfeatures[1]/n_electrodes
        f3=nfeatures[2]/n_electrodes
        f4=nfeatures[3]/n_electrodes
        f5=nfeatures[4]/n_electrodes
        f6=nfeatures[5]/n_electrodes
        f7=nfeatures[6]/n_electrodes
        f8=nfeatures[7]/n_electrodes
        f9=nfeatures[8]/n_electrodes
        f10=nfeatures[9]/n_electrodes
        f11=nfeatures[10]/n_electrodes
        f12=nfeatures[11]/n_electrodes
        f13=nfeatures[12]/n_electrodes
        f14=nfeatures[13]/n_electrodes
        f15=nfeatures[14]/n_electrodes
        f16=nfeatures[15]/n_electrodes
        f17=nfeatures[16]/n_electrodes
        f18=nfeatures[17]/n_electrodes
        f19=nfeatures[18]/n_electrodes
        f20=nfeatures[19]/n_electrodes
        f21=nfeatures[20]/n_electrodes
        f22=nfeatures[21]/n_electrodes
        f23=nfeatures[22]/n_electrodes
        f24=nfeatures[23]/n_electrodes
        f25=nfeatures[24]/n_electrodes
        f26=nfeatures[25]/n_electrodes
        f27=nfeatures[26]/n_electrodes
        f28=nfeatures[27]/n_electrodes
        f29=nfeatures[28]/n_electrodes
        f30=nfeatures[29]/n_electrodes
        f31=nfeatures[30]/n_electrodes
        f32=nfeatures[31]/n_electrodes
        f33=nfeatures[32]/n_electrodes
        f34=nfeatures[33]/n_electrodes
        f35=nfeatures[34]/n_electrodes
        f36=nfeatures[35]/n_electrodes
        


        f=[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36]


        def create_dictionary(keys):
            result = {} # empty dictionary
            for key in keys:
                result[key] = []
            return result

        keys=[]
        for i in range(0, data_train_loop.shape[1]):
            keys.append("e"+str(i))

        electrodes=create_dictionary(keys)
        #electrodes={"e0":[], "e1":[], "e2":[], "e3":[], "e4":[], "e5":[], "e6":[], "e7":[]}
        count=0
        llim=0
        for i in range(0, len(f)):
            count=0
            for j in range(llim,llim+nfeatures[i], int(f[i])):
                electrodes["e"+str(count)].append(np.sum(importances[j:j+int(f[i])]))
                #print(i)
                #print(count)
                count=count+1
            llim=llim+nfeatures[i]

        for e in electrodes:
            electrodes[str(e)]=sum(electrodes[str(e)])

        final_df = pd.DataFrame({"Electrodes": electrodes.keys(), "Importances":electrodes.values()})
        final_df["main"]=np.arange(data_train_loop.shape[1])
        temp_df=final_df.sort_values(by='Importances', ascending=False)
        topn=temp_df.values[:,2]
        print("topn electrodes :",topn)
        #print("topn electrodes datatype :", topn.dtype)

        keys=[]
        for i in range(0, data_train_loop.shape[1]):
            keys.append(str(i+1))

        accuracy=create_dictionary(keys)

        for i in range(0,data_train_loop.shape[1]):
            print("iteration"+str(i))

        l_feat=args.l_feat
        f_split=args.f_split
        acc=[]
        best_params=[]
        for i in range(data_train.shape[1]):
            if(i==0):
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[:,np.newaxis,i,:], data_test_loop[:,np.newaxis,i,:],f_split, 1000, l_feat, False) 
            else:
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[:,list(topn[0:i+1]),:], data_test_loop[:,list(topn[0:i+1]),:],f_split, 1000, l_feat, True)
            # Without feature selection check auuracy with Random forest
            if(args.classifier=="RF"):
                rf = RandomForestClassifier()
                distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                clf.fit(df_train_temp.values, labels_train_loop)
                y_pred_rf_w = clf.predict(df_test_temp.values)
                best_params.append(clf.best_params_)
                acc.append(metrics.accuracy_score(labels_test_loop,y_pred_rf_w))
            elif(args.classifier=="SVM"):
                object=StandardScaler()
                object.fit(df_train_temp)
                df_train_temp = object.transform(df_train_temp) 
                df_test_temp=object.transform(df_test_temp)  
                svma=svm.SVC()
                distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                clf.fit(df_train_temp, labels_train_loop)
                y_pred_rf_w = clf.predict(df_test_temp)
                best_params.append(clf.best_params_)
                acc.append(metrics.accuracy_score(labels_test_loop,y_pred_rf_w))

        sd=0

        return acc, best_params, sd

    elif(args.dataset=="speech"):
        data_ib=np.load('./data/speech.npz')
        data_train_ib = data_ib["X_Train"]
        labels_train_ib = data_ib["Y_Train"]
        #500 Tstep
        data_train_ib_500, rep=segment_speech(data_train_ib, segment_length=500)
        print(np.amax(data_train_ib_500))
        print(np.amin(data_train_ib_500))

        segment_length=500
        labels_train_ib_500=repeater(labels_train_ib, rep)

        #1000 Tstep
        data_train_ib_1000, rep=segment_speech(data_train_ib, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=repeater(labels_train_ib, rep)

        '''#1500 Tstep
        data_train_ib_1500=segment_speech(data_train_ib, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=repeater(labels_train_ib, rep)

        #3000 Tstep
        data_train_ib_3000=data_train_ib
        segment_length=3000
        labels_train_ib_3000=labels_train_ib'''

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000}
        segment_length=[500,1000]


        kf3 = StratifiedKFold(n_splits=5, shuffle=False)

        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]

        #fs=int(segment_length[i]/3)

        rf = RandomForestClassifier()
        '''distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
        clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)'''
        df_train, llim, nfeatures=createFV_individual_feat(data_train_loop,  1000, l_feat, True)
        rf.fit(df_train.values, labels_train_loop)
        
        # Without feature selection check auuracy with Random forest

        importances = rf.feature_importances_
        print(importances)

        n_electrodes=data_train_loop.shape[1]
        #dictionary of features
        #nfeatures=[]
        f1=nfeatures[0]/n_electrodes
        f2=nfeatures[1]/n_electrodes
        f3=nfeatures[2]/n_electrodes
        f4=nfeatures[3]/n_electrodes
        f5=nfeatures[4]/n_electrodes
        f6=nfeatures[5]/n_electrodes
        f7=nfeatures[6]/n_electrodes
        f8=nfeatures[7]/n_electrodes
        f9=nfeatures[8]/n_electrodes
        f10=nfeatures[9]/n_electrodes
        f11=nfeatures[10]/n_electrodes
        f12=nfeatures[11]/n_electrodes
        f13=nfeatures[12]/n_electrodes
        f14=nfeatures[13]/n_electrodes
        f15=nfeatures[14]/n_electrodes
        f16=nfeatures[15]/n_electrodes
        f17=nfeatures[16]/n_electrodes
        f18=nfeatures[17]/n_electrodes
        f19=nfeatures[18]/n_electrodes
        f20=nfeatures[19]/n_electrodes
        f21=nfeatures[20]/n_electrodes
        f22=nfeatures[21]/n_electrodes
        f23=nfeatures[22]/n_electrodes
        f24=nfeatures[23]/n_electrodes
        f25=nfeatures[24]/n_electrodes
        f26=nfeatures[25]/n_electrodes
        f27=nfeatures[26]/n_electrodes
        f28=nfeatures[27]/n_electrodes
        f29=nfeatures[28]/n_electrodes
        f30=nfeatures[29]/n_electrodes
        f31=nfeatures[30]/n_electrodes
        f32=nfeatures[31]/n_electrodes
        f33=nfeatures[32]/n_electrodes
        f34=nfeatures[33]/n_electrodes
        f35=nfeatures[34]/n_electrodes
        f36=nfeatures[35]/n_electrodes


        f=[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36]


        def create_dictionary(keys):
            result = {} # empty dictionary
            for key in keys:
                result[key] = []
            return result

        keys=[]
        for i in range(0, data_train_loop.shape[1]):
            keys.append("e"+str(i))

        electrodes=create_dictionary(keys)
        #electrodes={"e0":[], "e1":[], "e2":[], "e3":[], "e4":[], "e5":[], "e6":[], "e7":[]}
        count=0
        llim=0
        for i in range(0, len(f)):
            count=0
            for j in range(llim,llim+nfeatures[i], int(f[i])):
                electrodes["e"+str(count)].append(np.sum(importances[j:j+int(f[i])]))
                #print(i)
                #print(count)
                count=count+1
            llim=llim+nfeatures[i]

        for e in electrodes:
            electrodes[str(e)]=sum(electrodes[str(e)])

        final_df = pd.DataFrame({"Electrodes": electrodes.keys(), "Importances":electrodes.values()})
        final_df["main"]=np.arange(data_train_loop.shape[1])
        temp_df=final_df.sort_values(by='Importances', ascending=False)
        topn=temp_df.values[:,2]
        print("topn electrodes :",topn)
        #print("topn electrodes datatype :", topn.dtype)

        keys=[]
        for i in range(0, data_train_loop.shape[1]):
            keys.append(str(i+1))

        accuracy=create_dictionary(keys)

        for i in range(0,data_train_loop.shape[1]):
            print("iteration"+str(i))

        l_feat=args.l_feat
        acc={}
        best_params={}
        sd={}
        for i in range(data_train_loop.shape[1]):
            acc[str(i)]=[]
        for i in range(data_train_loop.shape[1]):
            best_params[str(i)]=[]
        for i in range(data_train_loop.shape[1]):
            sd[str(i)]=[]

        for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
            data_train=data_train_loop[train_index]
            data_test=data_train_loop[test_index]
            for i in range(data_train_loop.shape[1]):
                if(i==0):
                    df_train_temp, df_test_temp=createFV_individual(data_train[:,np.newaxis,i,:], data_test[:,np.newaxis,i,:], 1000, l_feat, False)
                
                else:
                    df_train_temp, df_test_temp=createFV_individual(data_train[:,list(topn[0:i+1]),:], data_test[:,list(topn[0:i+1]),:], 1000, l_feat, True)
                # Without feature selection check auuracy with Random forest
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop[train_index])
                    y_pred_rf_w = clf.predict(df_test_temp.values)
                    best_params[str(i)].append(clf.best_params_)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],y_pred_rf_w))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop[train_index])
                    y_pred_rf_w = clf.predict(df_test_temp)
                    best_params[str(i)].append(clf.best_params_)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],y_pred_rf_w))

        for i in range(data_train_loop.shape[1]):
            sd[str(i)]=np.std(acc[str(i)])
        return acc, best_params, sd


    else:
        data_ib=np.load('./data/'+args.dataset+'_epochs.npz')
        data_train_ib = data_ib["X"]
        labels_train_ib = data_ib["y"]
        l_feat=args.l_feat
        f_split=args.f_split
        #500 Tstep
        data_train_ib_500=segment(data_train_ib, segment_length=500)
        print(np.amax(data_train_ib_500))
        print(np.amin(data_train_ib_500))

        segment_length=500
        labels_train_ib_500=np.repeat(labels_train_ib,3000/int(segment_length))

        #1000 Tstep
        data_train_ib_1000=segment(data_train_ib, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=np.repeat(labels_train_ib,3000/int(segment_length))

        #1500 Tstep
        data_train_ib_1500=segment(data_train_ib, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=np.repeat(labels_train_ib,3000/int(segment_length))

        #3000 Tstep
        data_train_ib_3000=data_train_ib
        segment_length=3000
        labels_train_ib_3000=labels_train_ib

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
        segment_length=[500,1000,1500,3000]
        l_feat=args.l_feat 
        n_iter=args.niter


        kf3 = StratifiedKFold(n_splits=5, shuffle=False)
        #accd={}

        #print("iteration "+str(i))
        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]

        #fs=int(segment_length[i]/3)

        rf = RandomForestClassifier()
        '''distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
        clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)'''
        df_train, llim, nfeatures=createFV_individual_feat(data_train_loop,  1000, l_feat, True)
        rf.fit(df_train.values, labels_train_loop)
        
        # Without feature selection check auuracy with Random forest

        importances = rf.feature_importances_
        print(importances)

        n_electrodes=data_train_loop.shape[1]
        #dictionary of features
        #nfeatures=[]
        f1=nfeatures[0]/n_electrodes
        f2=nfeatures[1]/n_electrodes
        f3=nfeatures[2]/n_electrodes
        f4=nfeatures[3]/n_electrodes
        f5=nfeatures[4]/n_electrodes
        f6=nfeatures[5]/n_electrodes
        f7=nfeatures[6]/n_electrodes
        f8=nfeatures[7]/n_electrodes
        f9=nfeatures[8]/n_electrodes
        f10=nfeatures[9]/n_electrodes
        f11=nfeatures[10]/n_electrodes
        f12=nfeatures[11]/n_electrodes
        f13=nfeatures[12]/n_electrodes
        f14=nfeatures[13]/n_electrodes
        f15=nfeatures[14]/n_electrodes
        f16=nfeatures[15]/n_electrodes
        f17=nfeatures[16]/n_electrodes
        f18=nfeatures[17]/n_electrodes
        f19=nfeatures[18]/n_electrodes
        f20=nfeatures[19]/n_electrodes
        f21=nfeatures[20]/n_electrodes
        f22=nfeatures[21]/n_electrodes
        f23=nfeatures[22]/n_electrodes
        f24=nfeatures[23]/n_electrodes
        f25=nfeatures[24]/n_electrodes
        f26=nfeatures[25]/n_electrodes
        f27=nfeatures[26]/n_electrodes
        f28=nfeatures[27]/n_electrodes
        f29=nfeatures[28]/n_electrodes
        f30=nfeatures[29]/n_electrodes
        f31=nfeatures[30]/n_electrodes
        f32=nfeatures[31]/n_electrodes
        f33=nfeatures[32]/n_electrodes
        f34=nfeatures[33]/n_electrodes
        f35=nfeatures[34]/n_electrodes
        f36=nfeatures[35]/n_electrodes


        f=[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36]


        def create_dictionary(keys):
            result = {} # empty dictionary
            for key in keys:
                result[key] = []
            return result

        keys=[]
        for i in range(0, data_train_loop.shape[1]):
            keys.append("e"+str(i))

        electrodes=create_dictionary(keys)
        #electrodes={"e0":[], "e1":[], "e2":[], "e3":[], "e4":[], "e5":[], "e6":[], "e7":[]}
        count=0
        llim=0
        for i in range(0, len(f)):
            count=0
            for j in range(llim,llim+nfeatures[i], int(f[i])):
                electrodes["e"+str(count)].append(np.sum(importances[j:j+int(f[i])]))
                #print(i)
                #print(count)
                count=count+1
            llim=llim+nfeatures[i]

        for e in electrodes:
            electrodes[str(e)]=sum(electrodes[str(e)])

        final_df = pd.DataFrame({"Electrodes": electrodes.keys(), "Importances":electrodes.values()})
        final_df["main"]=np.arange(data_train_loop.shape[1])
        temp_df=final_df.sort_values(by='Importances', ascending=False)
        topn=temp_df.values[:,2]
        print("topn electrodes :",topn)
        #print("topn electrodes datatype :", topn.dtype)

        keys=[]
        for i in range(0, data_train_loop.shape[1]):
            keys.append(str(i+1))

        accuracy=create_dictionary(keys)

        for i in range(0,data_train_loop.shape[1]):
            print("iteration"+str(i))

        l_feat=args.l_feat
        acc={}
        best_params={}
        sd={}
        for i in range(data_train_loop.shape[1]):
            acc[str(i)]=[]
        for i in range(data_train_loop.shape[1]):
            best_params[str(i)]=[]
        for i in range(data_train_loop.shape[1]):
            sd[str(i)]=[]

        for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
            data_train=data_train_loop[train_index]
            data_test=data_train_loop[test_index]
            for i in range(data_train_loop.shape[1]):
                if(i==0):
                    df_train_temp, df_test_temp=createFV_individual(data_train[:,np.newaxis,i,:], data_test[:,np.newaxis,i,:], 1000, l_feat, False)
                
                else:
                    df_train_temp, df_test_temp=createFV_individual(data_train[:,list(topn[0:i+1]),:], data_test[:,list(topn[0:i+1]),:], 1000, l_feat, True)
                # Without feature selection check auuracy with Random forest
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop[train_index])
                    y_pred_rf_w = clf.predict(df_test_temp.values)
                    best_params[str(i)].append(clf.best_params_)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],y_pred_rf_w))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop[train_index])
                    y_pred_rf_w = clf.predict(df_test_temp)
                    best_params[str(i)].append(clf.best_params_)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],y_pred_rf_w))

        for i in range(data_train_loop.shape[1]):
            sd[str(i)]=np.std(acc[str(i)])
        return acc, best_params, sd
      
def individual(args):
    if(args.dataset=="bci3"):
        data=np.load('./data/bci_3.npz')
        data_train=data["X"]
        data_test=data["X_test"]
        labels=data['events']
        truelabels=np.loadtxt("./true_labels.txt", delimiter="/n")
        data_train_ib_500=segment(data_train, segment_length=500)
        segment_length=500
        labels_train_ib_500=np.repeat(labels,3000/int(segment_length))

        data_test_ib_500=segment(data_test, segment_length=500)
        segment_length=500
        labels_test_ib_500=np.repeat(truelabels,3000/int(segment_length))

        #1000 Tstep
        data_train_ib_1000=segment(data_train, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=np.repeat(labels,3000/int(segment_length))

        data_test_ib_1000=segment(data_test, segment_length=1000)
        segment_length=1000
        labels_test_ib_1000=np.repeat(truelabels,3000/int(segment_length))

        #1500 Tstep
        data_train_ib_1500=segment(data_train, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=np.repeat(labels,3000/int(segment_length))

        data_test_ib_1500=segment(data_test, segment_length=1500)
        segment_length=1500
        labels_test_ib_1500=np.repeat(truelabels,3000/int(segment_length))

        data_train_ib_3000=data_train
        segment_length=3000
        labels_train_ib_3000=labels

        data_test_ib_3000=data_test
        segment_length=3000
        labels_test_ib_3000=truelabels

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
        testing_data={'500':data_test_ib_500, '1000':data_test_ib_1000, '1500':data_test_ib_1500, '3000':data_test_ib_3000}
        label_data_test={'500':labels_test_ib_500, '1000':labels_test_ib_1000, '1500':labels_test_ib_1500, '3000':labels_test_ib_3000}
        segment_length=[500,1000,1500,3000]
        l_feat=args.l_feat 
        n_iter=args.niter
        f_split=args.f_split

        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]
        data_test_loop=testing_data[str(args.tstep)]
        labels_test_loop=label_data_test[str(args.tstep)]

        acc={}
        best_params_ie={}
        for k in range(data_train_loop.shape[1]):
            acc[str(k)]=[]
        for k in range(data_train_loop.shape[1]):
            best_params_ie[str(k)]=[]

        for i in range(data_train_loop.shape[1]):
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[:,np.newaxis,i,:], data_test_loop[:,np.newaxis,i,:],f_split, 1000, l_feat, False)
                print(np.amax(df_train_temp.values))
                print(np.amin(df_train_temp.values))
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop)
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp.values)
                    acc[str(i)].append(metrics.accuracy_score(labels_test_loop,predictions))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop)
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp)
                    acc[str(i)].append(metrics.accuracy_score(labels_test_loop,predictions))
            # Without feature selection check accuracy with Random forest    
        for k in range(data_train_loop.shape[1]):
            acc[str(k)]=sum(acc[str(k)])/len(acc[str(k)]) 


        n_features=args.nfeatures
        accf={}
        for k in range(n_features):
            accf[str(k)]=[]  
        
        for i in range(n_features):
            df_train_temp, df_test_temp=createFV_individual(data_train_loop, data_test_loop,f_split, 1000, [i], True)
            print(np.amax(df_train_temp.values))
            print(np.amin(df_train_temp.values))
            if(args.classifier=="RF"):
                rf = RandomForestClassifier()
                distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                clf.fit(df_train_temp.values, labels_train_loop)
                predictions = clf.predict(df_test_temp.values)
                accf[str(i)].append(metrics.accuracy_score(labels_test_loop,predictions))
            elif(args.classifier=="SVM"):
                object=StandardScaler()
                object.fit(df_train_temp)
                df_train_temp = object.transform(df_train_temp) 
                df_test_temp=object.transform(df_test_temp) 
                svma=svm.SVC()
                distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                clf.fit(df_train_temp, labels_train_loop)          
                predictions = clf.predict(df_test_temp)
                accf[str(i)].append(metrics.accuracy_score(labels_train_loop,predictions))
        # Without feature selection check accuracy with Random forest    
        for k in range(n_features):
            accf[str(k)]=sum(accf[str(k)])/len(accf[str(k)]) 
            
        sd=0
        sdf=0

        return acc, sd, accf, sdf, best_params_ie



    else:
        data_ib=np.load('./data/'+args.dataset+'_epochs.npz')
        data_train_ib = data_ib["X"]
        labels_train_ib = data_ib["y"]
        #500 Tstep
        data_train_ib_500=segment(data_train_ib, segment_length=500)
        print(np.amax(data_train_ib_500))
        print(np.amin(data_train_ib_500))

        segment_length=500
        labels_train_ib_500=np.repeat(labels_train_ib,3000/int(segment_length))

        #1000 Tstep
        data_train_ib_1000=segment(data_train_ib, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=np.repeat(labels_train_ib,3000/int(segment_length))

        #1500 Tstep
        data_train_ib_1500=segment(data_train_ib, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=np.repeat(labels_train_ib,3000/int(segment_length))

        #3000 Tstep
        data_train_ib_3000=data_train_ib
        segment_length=3000
        labels_train_ib_3000=labels_train_ib

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
        segment_length=[500,1000,1500,3000]


        kf3 = StratifiedKFold(n_splits=5, shuffle=False)
        #accd={}

        #print("iteration "+str(i))
        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]
        #fs=int(segment_length[i]/3)
        acc={}
        sd={}
        best_params_ie={}
        l_feat=args.l_feat
        n_iter=args.niter
        f_split=args.f_split
        #n_generations=40
        for k in range(data_train_loop.shape[1]):
            acc[str(k)]=[]
        for k in range(data_train_loop.shape[1]):
            sd[str(k)]=[] 
        for k in range(data_train_loop.shape[1]):
            best_params_ie[str(k)]=[]
        
        for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
            for i in range(data_train_loop.shape[1]):
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index,np.newaxis,i,:], data_train_loop[test_index,np.newaxis,i,:],f_split,1000, l_feat, False)
                print(np.amax(df_train_temp.values))
                print(np.amin(df_train_temp.values))
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop[train_index])
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp.values)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop[train_index])
                    best_params_ie[str(i)].append(clf.best_params_)
                    predictions = clf.predict(df_test_temp)
                    acc[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
            # Without feature selection check accuracy with Random forest   
        for k in range(data_train_loop.shape[1]):
            sd[str(k)]=np.std(acc[str(k)])
        '''for k in range(data_train_loop.shape[1]):
            acc[str(k)]=sum(acc[str(k)])/len(acc[str(k)])'''
            


        n_features=args.nfeatures
        accf={}
        
        sdf={}
        for k in range(n_features):
            accf[str(k)]=[]  
        for k in range(n_features):
            sdf[str(k)]=[]  
        for train_index, test_index in kf3.split(data_train_loop, labels_train_loop):
            for i in range(n_features):
                df_train_temp, df_test_temp=createFV_individual(data_train_loop[train_index], data_train_loop[test_index],f_split,1000, [i], True)
                print(np.amax(df_train_temp.values))
                print(np.amin(df_train_temp.values))
                if(args.classifier=="RF"):
                    rf = RandomForestClassifier()
                    distributions=dict(n_estimators=np.logspace(0, 3, 2*n_iter).astype(int))
                    clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp.values, labels_train_loop[train_index])                
                    predictions = clf.predict(df_test_temp.values)
                    accf[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
                elif(args.classifier=="SVM"):
                    object=StandardScaler()
                    object.fit(df_train_temp)
                    df_train_temp = object.transform(df_train_temp) 
                    df_test_temp=object.transform(df_test_temp) 
                    svma=svm.SVC()
                    distributions=dict(C=np.logspace(-3, 2, 2*n_iter), gamma=np.logspace(-3, 2, 2*n_iter))
                    clf = RandomizedSearchCV(svma, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)
                    clf.fit(df_train_temp, labels_train_loop[train_index])                
                    predictions = clf.predict(df_test_temp)
                    accf[str(i)].append(metrics.accuracy_score(labels_train_loop[test_index],predictions))
            # Without feature selection check accuracy with Random forest  
        for k in range(n_features):
            sdf[str(k)]=np.std(accf[str(k)])
        '''for k in range(n_features):
            accf[str(k)]=sum(accf[str(k)])/len(accf[str(k)])'''
        


        

        return acc, sd, accf, sdf, best_params_ie
      
  
        #sb="jc_mot"
    
def topn_feat(args):
    if args.dataset=="bci3":
        data=np.load('./data/bci_3.npz')
        data_train=data["X"]
        data_test=data["X_test"]
        labels=data['events']
        truelabels=np.loadtxt("./true_labels.txt", delimiter="/n")
        data_train_ib_500=segment(data_train, segment_length=500)
        segment_length=500
        labels_train_ib_500=np.repeat(labels,3000/int(segment_length))

        data_test_ib_500=segment(data_test, segment_length=500)
        segment_length=500
        labels_test_ib_500=np.repeat(truelabels,3000/int(segment_length))

        #1000 Tstep
        data_train_ib_1000=segment(data_train, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=np.repeat(labels,3000/int(segment_length))

        data_test_ib_1000=segment(data_test, segment_length=1000)
        segment_length=1000
        labels_test_ib_1000=np.repeat(truelabels,3000/int(segment_length))

        #1500 Tstep
        data_train_ib_1500=segment(data_train, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=np.repeat(labels,3000/int(segment_length))

        data_test_ib_1500=segment(data_test, segment_length=1500)
        segment_length=1500
        labels_test_ib_1500=np.repeat(truelabels,3000/int(segment_length))

        data_train_ib_3000=data_train
        segment_length=3000
        labels_train_ib_3000=labels

        data_test_ib_3000=data_test
        segment_length=3000
        labels_test_ib_3000=truelabels

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
        testing_data={'500':data_test_ib_500, '1000':data_test_ib_1000, '1500':data_test_ib_1500, '3000':data_test_ib_3000}
        label_data_test={'500':labels_test_ib_500, '1000':labels_test_ib_1000, '1500':labels_test_ib_1500, '3000':labels_test_ib_3000}
        segment_length=[500,1000,1500,3000]
        l_feat=args.l_feat 
        n_iter=args.niter

        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]
        data_test_loop=testing_data[str(args.tstep)]
        labels_test_loop=label_data_test[str(args.tstep)]

        df_train_temp, df_test_temp=createFV_individual(data_train_loop, data_test_loop, 1000, l_feat, True)
        print(np.amax(df_train_temp.values))
        print(np.amin(df_train_temp.values))
        # Without feature selection check accuracy with Random forest
        rf = RandomForestClassifier()
        '''distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
        clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)'''
        df_train, llim, nfeatures =createFV_individual_feat(data_train_loop,  1000, l_feat, True)
        
        return llim, nfeatures, data_train.shape[1]
      
    else:
        data_ib=np.load('./data/'+args.dataset+'_epochs.npz')
        data_train_ib = data_ib["X"]
        labels_train_ib = data_ib["y"]
        l_feat=args.l_feat
        #500 Tstep
        data_train_ib_500=segment(data_train_ib, segment_length=500)
        print(np.amax(data_train_ib_500))
        print(np.amin(data_train_ib_500))

        segment_length=500
        labels_train_ib_500=np.repeat(labels_train_ib,3000/int(segment_length))

        #1000 Tstep
        data_train_ib_1000=segment(data_train_ib, segment_length=1000)
        segment_length=1000
        labels_train_ib_1000=np.repeat(labels_train_ib,3000/int(segment_length))

        #1500 Tstep
        data_train_ib_1500=segment(data_train_ib, segment_length=1500)
        segment_length=1500
        labels_train_ib_1500=np.repeat(labels_train_ib,3000/int(segment_length))

        #3000 Tstep
        data_train_ib_3000=data_train_ib
        segment_length=3000
        labels_train_ib_3000=labels_train_ib

        training_data={'500':data_train_ib_500, '1000':data_train_ib_1000, '1500':data_train_ib_1500, '3000':data_train_ib_3000}
        label_data={'500':labels_train_ib_500, '1000':labels_train_ib_1000, '1500':labels_train_ib_1500, '3000':labels_train_ib_3000}
        segment_length=[500,1000,1500,3000]
        l_feat=args.l_feat 
        n_iter=args.niter


        kf3 = StratifiedKFold(n_splits=5, shuffle=False)
        #accd={}

        #print("iteration "+str(i))
        data_train_loop=training_data[str(args.tstep)]
        labels_train_loop=label_data[str(args.tstep)]

        #fs=int(segment_length[i]/3)

        rf = RandomForestClassifier()
        '''distributions=dict(n_estimators=np.logspace(0, 3, 400).astype(int))
        clf = RandomizedSearchCV(rf, distributions, random_state=0, n_iter=n_iter,n_jobs=-1)'''
        df_train, llim, nfeatures=createFV_individual_feat(data_train_loop,  1000, l_feat, True)
        
        return llim, nfeatures, data_train_ib.shape[1]



        

    #final_df = pd.DataFrame({"N": accuracy.keys(), "Accuracy":accuracy.values()})

    

