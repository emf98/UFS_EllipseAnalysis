# # This file uses a pre-determined RF model architecture to conduct feature selection and calc skill scores.
# File generated 12/2/2025, E. Fernandez

#The definition statement/function requires:
    # number of cross validations (n)
    # day length for each year, aka indication of leadtime (idx)
    # input data array
    # output data array
    # climo array
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

from SkillStats_MOD import BSS, RAS, PAS

#_____________________________________________________________________#
#collect temp files.
def temp_files(shift,region,region_type,idx):
    #load output data file for TRAINING AND VALIDATION
    infile = open(f'./old_data/{str(region)}_anomtemps_reduced.p',"rb",)
    temp = pickle.load(infile) 
    temp = temp.reshape(62,149)
    infile.close()
    
    # load climo data
    infile = open(f'./old_data/{str(region)}_climoprob_reduced.p',"rb",)
    climo = pickle.load(infile) 
    climo = climo.reshape(62,149,2)
    infile.close()
    
    ##reshape temp files again to introduce lag. 
    f_output = temp[:52,shift:].reshape(52*idx)
    #print("Check temp shape:", f_output.shape)
    
    ##reshaping climo
    f_climo = climo[:52,shift:,:].reshape(52*idx,2)
    #print("Check climo shape:", f_climo.shape)

    #load output data file for TESTING
    infile = open(f'./new_data/2classtemps_{str(region_type)}.p',"rb",)
    TEST_temp = pickle.load(infile) 
    infile.close()
    
    # load climo data
    infile = open(f'./new_data/2classclimo_{str(region_type)}.p',"rb",)
    TEST_climo = pickle.load(infile) 
    infile.close()

    return f_output, f_climo, TEST_temp, TEST_climo;

#_____________________________________________________________________#
#conduct feature selection

def rf_featselect(n,shift,idx, train_input, test_input, region1, region_type):

    ##collect temp files using def above.
    train_output, train_climo, test_output, test_climo = temp_files(shift, region1,region_type,idx)
    
    X_train = train_input.iloc[:,:]
    X_test = test_input.iloc[:,:]
    Y_train = train_output[:]
    Y_test = test_output[:]

    val_subset = (10*idx) 

    #empty lists to save Accuracy
    acc_reg1_val = []
    acc_reg1_train = []
    acc_reg1_test = []

    important = np.empty((n,7))

    #save probabilities
    p_test = []
    p_train = []
    p_val = []

    #create initial regressor for rf to do feature selection 
    rf_reg1 = RandomForestClassifier(max_depth=3, n_estimators=400, random_state=42)
    
    ##one-hot encoded outputs for the purpose of calculating probabilities
    #Y_all = keras.utils.to_categorical(output)
    #Y_tes = keras.utils.to_categorical(test_output)
    #X_all = input.values
    
    print("Begin CV ...")
    ##make loop for cross validation 
    for l in range(0,n):
        #print("Cross Val #:"+str(l))
        ##randomly choose a fraction of events for validation and training
        start = random.randrange(len(X_train.iloc[:,0])-val_subset)
        end = start+(val_subset)

        climo_test = test_climo[:,:]
        climo_tr = train_climo[:,:]

        X_val = X_train.iloc[start:end,:]
        Y_val = train_output[start:end]
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
        #pred1 = rf_reg1.predict(X_all)
        #pred2 = rf_reg1.predict_proba(X_all)

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
        selected_features = test_input.columns[selector.get_support()]
        #print(f'Selected Features: {selected_features}')

        importances = rf_reg1.feature_importances_
        important[l,:] = importances[:]

    
    print('###################################################')
    print(f'Accuracy, Validation: {np.mean(acc_reg1_val) * 100:.2f}%')
    print(f'Accuracy, Training: {np.mean(acc_reg1_train) * 100:.2f}%')
    print(f'Accuracy, Testing: {np.mean(acc_reg1_test) * 100:.2f}%')
    
    return important;

#_________________________________________________________________________#
##definition statement for ACC
def calculate_accuracy(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return np.mean(y_true == y_pred)

##definition for RF loop with confidence evaluation
def rf_90thpercentile(n,shift,idx, train_input, test_input, era5_truth, region1, region_type):

    ##collect temp files using def above.
    train_output, train_climo, test_output, test_climo = temp_files(shift, region1,region_type,idx)
    
    X_train = train_input.iloc[:,:]
    X_test = test_input.iloc[:,:]
    Y_train = train_output[:]
    Y_test = test_output[:]

    ERA5_truth = era5_truth[:]

    val_subset = (10*idx) 
    #______________________________________________________#
    print(" ")
    print("Creating lists ...")
    #empty lists to save Accuracy
    acc_reg2_val = []
    acc_reg2_train = []
    
    acc_reg2_test = [] ##testing versus its own truth
    acc_reg3_test = []##testing versus era5 truth
    acc_reg4_test = [] ##testing truth vs era5 truth

    ##BSS Arrays
    ##BSS Arrays, all of the skill scores have 200 rows
    #because that is how many cross-validations I will do for the model
    BSS_val= np.empty((n,))
    BSS_train= np.empty((n,))
    BSS_test= np.empty((n,))

    ##RAS and PAS Arrays
    Prec_val= np.empty((n,2))
    Rec_val= np.empty((n,2))

    Prec_train= np.empty((n,2))
    Rec_train= np.empty((n,2))

    Prec_test= np.empty((n,2))
    Rec_test= np.empty((n,2))

    #save PREDICTIONS
    test90_acc = []
    ##full model
    fulltest_acc = []

    ##correct positive
    posXtest = []
    #false positive
    FposXtest = []
    #correct negative
    negXtest = []
    #false negative
    FnegXtest = []

    indexes = []

    ##correct positive
    percpos = []
    #false positive
    percFpos = []
    #correct negative
    percneg = []
    #false negative
    percFneg = []
    
    #_______________________________________________________#
    print(" ")
    print("Establishing RF hyperparams ...")
    #second random forest model with selected features only
    rf_reg2 = RandomForestClassifier(max_depth=3, n_estimators=400, random_state=42)
    Y_tes = keras.utils.to_categorical(test_output)

    ##make loop for cross validation 
    print(" ")
    print("Begin RF CV.")
    for l in range(0,n):
        #print("Cross Val #:"+str(l))
        ##randomly choose a fraction of events for validation and training
        start = random.randrange(len(X_train.iloc[:,0])-val_subset)
        end = start+(val_subset)
        
        climo_test = test_climo[:,:]
        climo_tr = train_climo[:,:]

        X_val = X_train.iloc[start:end,:]
        Y_val = train_output[start:end]
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
        #_______________________train the model_______________________________
        #train rf
        rf_reg2.fit(X_tr, Y_tr)
       
        #prediction with validation data
        pred_val1 = rf_reg2.predict(X_val)
        pred_val2 = rf_reg2.predict_proba(X_val)
        #p_val.append(pred_val2)
        acc_reg2_val.append(accuracy_score(Y_val, pred_val1))

        #prediction with training data
        pred_train1 = rf_reg2.predict(X_tr)
        pred_train2 = rf_reg2.predict_proba(X_tr)
        #p_train.append(pred_train2)
        acc_reg2_train.append(accuracy_score(Y_tr, pred_train1))

        #prediction with testing data
        pred_test1 = rf_reg2.predict(X_test)
        pred_test2 = rf_reg2.predict_proba(X_test)
        
        #p_test.append(pred_test2)
        acc_reg2_test.append(accuracy_score(Y_test, pred_test1))
        acc_reg3_test.append(accuracy_score(ERA5_truth, pred_test1)) 
        acc_reg4_test.append(accuracy_score(ERA5_truth, Y_test))

        #_______________________statistics calcs_______________________________
        pred_class = []
        predval_class = []
        predtr_class = []
        predtest_class = []

        Y_tr2 = keras.utils.to_categorical(Y_tr)
        Y_val2 = keras.utils.to_categorical(Y_val)

        ##BRIER SKILL SCORE
        BSS_val[l] = BSS(Y_val2,pred_val2)
        BSS_train[l] = BSS(Y_tr2,pred_train2)
        BSS_test[l] = BSS(Y_tes,pred_test2) 

        ##RECALL ACCURACY SCORE    
        RAS(l,climo_val, Rec_val, Y_val2, pred_val2, predval_class,
                climo_train, Rec_train, Y_tr2, pred_train2, predtr_class,
                climo_test, Rec_test, Y_tes, pred_test2, predtest_class)
        ##PRECISION ACCURACY SCORE     
        PAS(l, climo_val, Prec_val, Y_val2, pred_val2, predval_class,
                climo_train, Prec_train, Y_tr2, pred_train2, predtr_class,
                climo_test, Prec_test, Y_tes, pred_test2, predtest_class)

        #___________________________Higher confidence samples_______________________

        q90 = np.percentile(pred_test2,90,axis=0) ##90th percentile of test
        ##90th percentile acc
        great90 = [i for i, row in enumerate(pred_test2) if (row[0] > q90[0]) or (row[1] > q90[1])]
        # Create the arrays of probabilities and actual values that exceed the 90th percentile
        test90 = pred_test2[great90]
        test90_norm = Y_tes[great90]
        test90_acc.append(calculate_accuracy(test90_norm, test90, threshold=0.5))
        ##full model
        fulltest_acc.append(calculate_accuracy(Y_tes, pred_test2, threshold=0.5))

        ##classify the accuracy of predicitons
        correct_pos = [] #correct positive anomaly
        correct_neg = [] #correct negative anomaly

        false_pos = [] #falsely positive
        false_neg = [] #falsely negative

        indexes.extend(great90)
        for j in range(len(great90)):
            index = great90[j]

            if pred_test2[index, 1] > pred_test2[index, 0] and Y_tes[index, 1] == 1:
                correct_pos.append(index)
            elif pred_test2[index, 0] > pred_test2[index, 1] and Y_tes[index, 0] == 1:
                correct_neg.append(index)
            elif pred_test2[index, 1] > pred_test2[index, 0] and Y_tes[index, 0] == 1:
                false_pos.append(index)
            elif pred_test2[index, 0] > pred_test2[index, 1] and Y_tes[index, 1] == 1:
                false_neg.append(index)

        ##correct positive
        posXtest.extend(correct_pos)
        #false positive
        FposXtest.extend(false_pos)
        #correct negative
        negXtest.extend(correct_neg)
        #false negative
        FnegXtest.extend(false_neg)

        percpos.append(len(correct_pos)/len(great90))
        #false positive
        percFpos.append(len(false_pos)/len(great90))
        #correct negative
        percneg.append(len(correct_neg)/len(great90))
        #false negative
        percFneg.append(len(false_neg)/len(great90))
        
    print(" ")
    print("END CV.")
    print(" ")
    print(" ")
    print('###################################################')
    print(f'Accuracy, Validation: {np.mean(acc_reg2_val) * 100:.2f}%')
    print(f'Accuracy, Training: {np.mean(acc_reg2_train) * 100:.2f}%')
    print(f'Accuracy, Testing: {np.mean(acc_reg2_test) * 100:.2f}%')
    
    ## testing print avg BSS/RAS/PAS
    print("_____________________________________________________________________")
    print(f"Brier Skill Score (Train): {np.nanmean(BSS_train,axis=0):.4f}")
    print(f"Brier Skill Score (Test): {np.nanmean(BSS_test,axis=0):.4f}")
    print(f"Brier Skill Score (Validation): {np.nanmean(BSS_val,axis=0):.4f}")
    print("_____________________________________________________________________")
    print("Recall and Precision: Neg Cat")
    print("#########")
    print(f"Recall Accuracy Score (Train): {np.nanmean(Rec_train[:,0],axis=0):.4f}")
    print(f"Recall Accuracy Score (Test): {np.nanmean(Rec_test[:,0],axis=0):.4f}")
    print(f"Recall AccuracyScore (Validation): {np.nanmean(Rec_val[:,0],axis=0):.4f}")
    print(f"Precision Accuracy Score (Train): {np.nanmean(Prec_train[:,0],axis=0):.4f}")
    print(f"Precision Accuracy Score (Test): {np.nanmean(Prec_test[:,0],axis=0):.4f}")
    print(f"Precision AccuracyScore (Validation): {np.nanmean(Prec_val[:,0],axis=0):.4f}")
    print("_____________________________________________________________________")
    print("Recall and Precision: Pos Cat")
    print("#########")
    print(f"Recall Accuracy Score (Train): {np.nanmean(Rec_train[:,1],axis=0):.4f}")
    print(f"Recall Accuracy Score (Test): {np.nanmean(Rec_test[:,1],axis=0):.4f}")
    print(f"Recall AccuracyScore (Validation): {np.nanmean(Rec_val[:,1],axis=0):.4f}")
    print(f"Precision Accuracy Score (Train): {np.nanmean(Prec_train[:,1],axis=0):.4f}")
    print(f"Precision Accuracy Score (Test): {np.nanmean(Prec_test[:,1],axis=0):.4f}")
    print(f"Precision AccuracyScore (Validation): {np.nanmean(Prec_val[:,1],axis=0):.4f}")
    
    ##correct positive
    posXtest = np.array(posXtest)
    #false positive
    FposXtest = np.array(FposXtest)
    #correct negative
    negXtest = np.array(negXtest)
    #false negative
    FnegXtest = np.array(FnegXtest)

    indexes = np.array(indexes)
    print("_____________________________________________________________________")
    print(f'Average Num. of 10% Confident and Correct Postive Predictions: {np.mean(percpos)* 100:.2f}%')
    print(f'Average Num. of 10% Confident and Correct Negative Predictions: {np.mean(percneg)* 100:.2f}%')
    print(f'Average Num. of 10% Confident and FALSE Postive Predictions: {np.mean(percFpos)* 100:.2f}%')
    print(f'Average Num. of 10% Confident and FALSE Negative Predictions: {np.mean(percFneg)* 100:.2f}%')
    print('#######################################################################')
    print(f'Average Num. of 10% Confident and Correct Predictions: {np.mean(percpos)* 100 + np.mean(percneg)* 100:.2f}%')
    print(f'Average Num. of 10% Confident and FALSE Predictions: {np.mean(percFpos)* 100 +np.mean(percFneg)* 100:.2f}%')

    #try SHAP with this model
    print(" ")
    print(" ")
    print("Trying SHAP ...")
    import shap
    
    explainer = shap.TreeExplainer(rf_reg2)
    shap_values = explainer.shap_values(X_test)
    shap_obj = explainer(X_test) ##return this value to plot. 
    print("Done.")
    return test90_acc, fulltest_acc, shap_obj, posXtest, FposXtest, negXtest, FnegXtest, acc_reg2_test, acc_reg3_test, acc_reg4_test;

