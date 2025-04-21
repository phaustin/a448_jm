# %%
# IMPORTANT
# This file tunes the hyperparameters of the Random Forest using a weighting method to mitigate the class imbalance
# in working directory: X_train, y_train folder, coord_dict.pkl (this is for subsetting the domain)
# set smoke_test = True for short run. False for actual run
# the paths in the retreive_X_y_train function need to be changed to get the training data
# the output will be a log file labeled tuningW.log to monitor the progress (if the progress is too slow, I can change the code), and a csv file labeled tunedW.csv


import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from time import time 
import logging
import imblearn
import xarray as xr
import csv
import pickle 
import joblib
from memory_profiler import profile
import itertools
# %%
def retreive_X_y_train(dates,feature_lis,shape):
    times = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    y_lis = []
    X_lis = []
    for date in dates:
        y = np.load('preprocessed/full_domain/y_train/y_train'+date[0]+date[1]+date[2]+'.npy')
        X = np.load('preprocessed/feature_subset/X_train/X_train'+date[0]+date[1]+date[2]+'.npy')
        y_sub, X_sub = subset_X_y(X,y,date,times,feature_lis,shape)
        y_lis.append(y_sub)
        X_lis.append(X_sub)
    
    y = np.concatenate(y_lis)
    X = np.concatenate(X_lis)

    return y, X

def subset_X_y(X_full,y_full,date,times,feature_lis,shape1):
    """""
    This function subsets the X and y data to obtain only the domain of overlap between the WRF 12km and GLM domains
    This is done by retreiving the saved dictionary coord_dict.pkl which contains the index (tuple) of each gridcell as keys and a
    random variable where the out-of-bounds region is filled with -999 as the values. 
    In: y and X matrices for the full domain
    Out: y and X matrices for the subsetted domain
    Note: When reconstructing the results (model output), use the dictionary with the saved corrdinates.
    """""

    # Retreive dictionary containing coordinates and random variable with fill values 
    with open('coord_dict.pkl', 'rb') as f:
        coord_dict = pickle.load(f)

    # Extract corrdinates and random variable with fill values from dictionary as 1d arrays
    coords = np.array(list(coord_dict.keys()))
    random_vals = np.array(list(coord_dict.values()))

    y_lis_final = []
    X_lis_final = []
    for i,tt in enumerate(times):
        y_train = y_full[i*shape1[0]*shape1[1]:i*shape1[0]*shape1[1]+shape1[0]*shape1[1]]
        X_train = X_full[i*shape1[0]*shape1[1]:i*shape1[0]*shape1[1]+shape1[0]*shape1[1],:]

        # replace the out of bounds values with fill values 
        y_train_fill = np.where(random_vals==-999,-999,y_train)

        # Remove the fill values from the array
        y_new = y_train_fill[y_train_fill != -999]

        y_lis_final.append(y_new)
        X_lis = []
        for xx,var in enumerate(feature_lis):
            X_train_var = X_train[:,xx]

            # replace the out of bounds values with fill values 
            X_train_fill = np.where(random_vals==-999,-999,X_train_var)

            # Remove the fill values from the array
            X_new = X_train_fill[X_train_fill != -999]

            X_lis.append(X_new)
        X_lis_final.append(X_lis)
        
    y_sub = np.concatenate(y_lis_final,axis=0)
    X_sub = np.concatenate(X_lis_final,axis=1).T

    return y_sub,X_sub

def calc_LogLoss(estimator, X, y):
    """
    Calculate the Log Loss for the RandomForrestClassifier and the LinearRegression model. Lower values indicate better performance.

    Parameters
    ----------
    estimator : sklearn.linear_model or sklearn.ensemble
        SkLearn ML model
    X : np.ndarray
        X matrix of predictors
    y : np.ndarray
        y vector of observations

    Returns
    ------
    LogLoss : float
        The log loss 
    """
    try:
        # for the RF
        proba = estimator.predict_proba(X)[:,1]
    except AttributeError:
        # for the LR
        proba = estimator.predict(X)

    # since log at 0 is -inf, we add a tiny offset to prevent the equation from blowing up.
    epsilon = 1e-15
    p = np.clip(proba, epsilon, 1-epsilon)

    N = len(y)
    penalty = (y*np.log10(p) + (1-y)*np.log10(1-p))
    LogLoss = -1/N * np.sum(penalty)

    return LogLoss

def tune_hyperparameters(dict_hyper,X_train_all, y_train_bi,k):
    """
    Tune the hyperparameters for two random forests on the original and upsampled data. Both RFs are tested on the SAME subset of data.
    tuned.csv in the working directory is the csv file of all the different hyperparameter combinations and their Brier score and Brier skill score. 
    tuning.log tracks the progress of the hyperparameter tuning. 

    Parameters
    ----------
    dict_hyperparam : dictionary
        keys=names of hyperparameters (they must correspond to the Scikit Learn hyperparameter names)
        values=list of each hyperparameter

    """
    sk_folds = StratifiedKFold(n_splits = k, shuffle = False)

    keys = dict_hyper.keys()
    fields = list(keys) + ['LL_RF_w','LL_LR']
    itterations = 1
    for value in dict_hyper.values():
        itterations*=len(value)

    logging.basicConfig(filename='tuningW1.log', level=logging.INFO)
    with open('tunedW1.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)

        i = 0 # to count itterations
        for combinations in itertools.product(*dict_hyper.values()):
            # the itertools.product function is like creating x nested for loops (where x is the number of keys in the dictionary)
            # this method is cleaner and more generalizable than hardcoding the for loops 
            dict_hyper_i = dict(zip(keys,combinations))

            i += 1
            start = time()
            logging.info(f'------------------------------------------------------------')
            logging.info(f'Itteration number: {i} of {itterations}. {round(i/itterations*100,1)}% complete')
            logging.info(f'Hyperparameters: {dict_hyper_i}')

            # initialize the models 
            rf = RandomForestClassifier(bootstrap=True, 
                                        max_samples=dict_hyper_i['max_samples'], 
                                        n_estimators=dict_hyper_i['n_estimators'], 
                                        max_depth=dict_hyper_i['max_depth'], 
                                        criterion=dict_hyper_i['criterion'],
                                        class_weight = dict_hyper_i['class_weight'],
                                        verbose=dict_hyper_i['verbose'])
            lr = LinearRegression()

            # perform the stratified k-fold cross-validation
            LL_rf_w_lis = []
            LL_lr_lis = []
            ki = 0
            for train_ii, test_ii in sk_folds.split(X_train_all, y_train_bi):
                ki +=1
                if ki==1:
                    logging.info(f'K-fold: {ki}')

                    # index all the data according to the split indices
                    X_train = X_train_all[train_ii]
                    X_test = X_train_all[test_ii]
                    y_train = y_train_bi[train_ii]
                    y_test = y_train_bi[test_ii]

                    # fit the models
                    rf_w_fit = rf.fit(X_train,y_train) # RF with original data 
                    lr_fit = lr.fit(X_train,y_train) # linear regression with original data

                    # save the model
                    joblib.dump(rf_w_fit, open(f'./models/RFw{i}.pkl','wb'), 9)

                    # test the models to get the log loss (note that they are all tested on the SAME data)
                    LL_rf_w = calc_LogLoss(rf_w_fit,X_test,y_test)
                    LL_lr = calc_LogLoss(lr_fit,X_test,y_test)

                    # append to lists
                    LL_rf_w_lis.append(LL_rf_w)
                    LL_lr_lis.append(LL_lr)
                else:
                    continue
            
            # take average score over the k folds
            LL_RF_w = np.mean(LL_rf_w_lis)
            LL_LR = np.mean(LL_lr_lis)

            logging.info(f'LL_RF_w: {LL_RF_w}, LL_LR: {LL_LR}')
            end = time()
            logging.info(f'Time for itteration: {(end-start)/60/60} hours')

            # add the hyperparameters and the scores to the tuned.csv file 
            lis_tuned = list(combinations) + [LL_RF_w,LL_LR]
            writer.writerow(lis_tuned)

        return
# %%
times = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
feature_lis = ['pw','cape','cin','T2','slp','ctt','K','SWI','LI','SI','wmax','Q']
shape = (417, 627)

smoke_test = False
if smoke_test:
    training_dates = [('2024','06','05')]
else:
    training_dates = [('2024','06','05'),('2024','06','06'),('2024','06','07'),('2024','06','09'),('2024','06','10'),('2024','06','11'),('2024','06','12'),('2024','06','13'),('2024','06','14'),('2024','06','15'),('2024','06','16'),('2024','06','18'),('2024','06','19'),('2024','06','20'),('2024','06','21'),('2024','07','14'),('2024','07','15'),('2024','07','16'),('2024','07','17'),('2024','07','19'),('2024','07','20'),('2024','07','21'),('2024','07','22'),('2024','07','23'),('2024','07','24'),('2024','07','25'),('2024','07','26'),('2024','07','27')]

y_train_all, X_train_all = retreive_X_y_train(training_dates,feature_lis,shape)
y_train_bi = np.where(y_train_all>1,1,0)

# number of folds
k = 3

n_feature = X_train_all.shape[1]
n_samples = X_train_all.shape[0]
max_samples = [int(n_samples/20)] # max num of samples to draw from training data to train each tree (default = num of rows of training set)
n_estimators = [200] # max num of trees (default = 100)
max_depth = [n_feature, 50, None] # max depth of trees (first is num of features, None causes nodes to expand until all leaves are pure or until 2 samples per leaf)
criterion = ['gini'] # function to measure the quality of a split
class_weight = ['balanced_subsample']
verbose = [9]

dict_hyper = {'max_samples':max_samples, 'n_estimators':n_estimators, 'max_depth':max_depth, 'criterion':criterion, 'class_weight':class_weight, 'verbose':verbose}

tune_hyperparameters(dict_hyper,X_train_all, y_train_bi,k)
# %%
