# # This file uses a pre-determined RF model architecture to conduct feature selection.
# File generated 9/18/2025, E. Fernandez

#The definition statement/function requires:
    # number of cross validations (n)
    # day length for each year, aka indication of leadtime (idx)
    # input data array
    # output data array
    # climo array

#The list "important" is returned so that the user can make the feature importance bar plot. 
#____________________________________________________________________#
#relevant import statements
import numpy as np
import math
import pandas as pd
import xarray as xr 
import pickle 
import matplotlib.pyplot as plt

import keras

import random
from random import seed
from random import randint
from random import sample

#import seaborn as sns # statistical data visualization
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.feature_selection import SelectFromModel
##just to stop the excess number of warnings 
import warnings
warnings.filterwarnings('ignore')

# #in order to make my life a little easier for the leadtime analysis and different regions, making this task into a function will save me a lot of space in notebooks. 

# the goal of this python file is the RF architecture for FEATURE SELECTION

def rf_featselect(n,idx,input,output,climo):
    
    X_train = input.iloc[:(52*idx),:]
    X_test = input.iloc[(52*idx):,:]
    Y_train = output[:(52*idx)]
    Y_test = output[(52*idx):]

    val_subset = (10*idx) 

    #empty lists to save Accuracy
    acc_reg1_val = []
    acc_reg1_train = []
    acc_reg1_test = []

    important = np.empty((n,8))

    #save probabilities
    p_test = []
    p_train = []
    p_val = []

    #create initial regressor for rf to do feature selection 
    rf_reg1 = RandomForestClassifier(max_depth=3, n_estimators=400, random_state=42)
    
    ##one-hot encoded outputs for the purpose of calculating probabilities
    Y_all = keras.utils.to_categorical(output)
    Y_tes = keras.utils.to_categorical(Y_test)
    X_all = input.values
    
    print("Begin CV ...")
    ##make loop for cross validation 
    for l in range(0,n):
        #print("Cross Val #:"+str(l))
        ##randomly choose a fraction of events for validation and training
        start = random.randrange(len(X_train.iloc[:,0])-val_subset)
        end = start+(val_subset)

        climo_test = climo[(52*idx):,:]
        climo_tr = climo[:(52*idx),:]

        X_val = X_train.iloc[start:end,:]
        Y_val = output[start:end]
        climo_val = climo_tr[start:end,:]

        X_train1 = X_train.iloc[0:start]
        Y_train1 = Y_train[0:start]
        climo_train1 = climo_tr[0:start,:]
        X_train2 = X_train.iloc[end:]
        Y_train2 = Y_train[end:]
        climo_train2 = climo_tr[end:,:]

        ##concatenate all of these
        X_tr = pd.concat([X_train1,X_train2], axis = 0)
        Y_tr = np.concatenate((Y_train1,Y_train2))
        climo_train = np.concatenate((climo_train1,climo_train2))


        #_______________________train the model_______________________________
        #train rf
        rf_reg1.fit(X_tr, Y_tr)
        pred1 = rf_reg1.predict(X_all)
        pred2 = rf_reg1.predict_proba(X_all)

        #prediction with validation data
        pred_val1 = rf_reg1.predict(X_val)
        pred_val2 = rf_reg1.predict_proba(X_val)
        #p_val.append(pred_val2)
        acc_reg1_val.append(accuracy_score(Y_val, pred_val1))

        #prediction with training data
        pred_train1 = rf_reg1.predict(X_tr)
        pred_train2 = rf_reg1.predict_proba(X_tr)
        #p_train.append(pred_train2)
        acc_reg1_train.append(accuracy_score(Y_tr, pred_train1))

        #prediction with testing data
        pred_test1 = rf_reg1.predict(X_test)
        pred_test2 = rf_reg1.predict_proba(X_test)
        #p_test.append(pred_test2)
        acc_reg1_test.append(accuracy_score(Y_test, pred_test1))

        #_______________________feature selection______________________________
        #prepare to show relevant features by actually ... choosing them 
        selector = SelectFromModel(rf_reg1, threshold="mean", max_features=None)
        X_train_selected = selector.transform(X_tr)
        X_test_selected = selector.transform(X_test)

        #print names of selected features
        selected_features = input.columns[selector.get_support()]
        #print(f'Selected Features: {selected_features}')

        importances = rf_reg1.feature_importances_
        important[l,:] = importances[:]

    
    print('###################################################')
    print(f'Accuracy, Validation: {np.mean(acc_reg1_val) * 100:.2f}%')
    print(f'Accuracy, Training: {np.mean(acc_reg1_train) * 100:.2f}%')
    print(f'Accuracy, Testing: {np.mean(acc_reg1_test) * 100:.2f}%')
    
    return important;
