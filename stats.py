
import numpy as np
from run import make_models, make_decision, retreive_model, get_wrf, get_X_y
from wrf import getvar, to_np, getvar, latlon_coords, get_basemap
from sklearn.metrics import roc_curve, roc_auc_score
from time import time 
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import pickle 
import imageio as io
import imblearn

def contingency_table(y_hat, y):
    """""
    In: y_hat = model prediction, y = observations
    Out = true positives, false positive, false negatives, true negatives as ditionary
    """""
    TP_KEY = 'num_true_positives'
    FP_KEY = 'num_false_positives'
    FN_KEY = 'num_false_negatives'
    TN_KEY = 'num_true_negatives'

    reg_binary = np.where(y_hat>=1, 1,0).flatten() # is 1 when there is 1 or more strikes, zero when less than 1 strike
    glm_binary = np.where(y>=1, 1,0).flatten()

    true_pos_indices = np.where(np.logical_and(reg_binary==1,glm_binary==1))[0]
    false_pos_indices = np.where(np.logical_and(reg_binary==1,glm_binary==0))[0]
    false_neg_indices = np.where(np.logical_and(reg_binary==0,glm_binary==1))[0]
    true_neg_indices = np.where(np.logical_and(reg_binary==0,glm_binary==0))[0]

    continge_dict = {TP_KEY: len(true_pos_indices), FP_KEY: len(false_pos_indices), FN_KEY: len(false_neg_indices), TN_KEY: len(true_neg_indices)}

    return continge_dict

def calc_CSI(y_hat,y):

    continge_dict = contingency_table(y_hat, y)
    CSI = continge_dict['num_true_positives'] / (continge_dict['num_true_positives'] + continge_dict['num_false_positives'] + continge_dict['num_false_negatives'])

    return CSI

def calc_TPR(y_hat,y):

    continge_dict = contingency_table(y_hat, y)
    TPR = continge_dict['num_true_positives'] / (continge_dict['num_true_positives'] + continge_dict['num_false_negatives'])
    
    return TPR

def calc_FPR(y_hat,y):

    continge_dict = contingency_table(y_hat, y)
    FPR = continge_dict['num_false_positives'] / (continge_dict['num_false_positives'] + continge_dict['num_true_negatives'])
    
    return FPR

def calc_deterministic_verification(p, y_test):

    threshs = np.linspace(0,1,200) # defines thresholds between 0 and 1 (default is 0.5)

    TPR_arr, FPR_arr, CSI_arr = np.zeros(len(threshs)), np.zeros(len(threshs)), np.zeros(len(threshs))
    
    for i,t in enumerate(threshs): 
        #find where the prediction is greater than or equal to the threshold
        p_bi = np.where(p >= t,1,0)

        try:
            #find statistics for the predictions at this specific threshold
            TPR = calc_TPR(p_bi, y_test)
            FPR = calc_FPR(p_bi, y_test)
            CSI = calc_CSI(p_bi, y_test)
        except ZeroDivisionError:
            break

        TPR_arr[i] = TPR
        FPR_arr[i] = FPR
        CSI_arr[i] = CSI

    # calculate AUC ( were flipping the arrays because they have to be in ascending order for the integration)
    AUC = np.trapz(np.flip(TPR_arr),np.flip(FPR_arr))

    return TPR_arr, FPR_arr, threshs, AUC, CSI_arr

def calc_LogLoss(proba,y):
    """
    Calculate the Log Loss for the RandomForrestClassifier. Lower values indicate better performance.

    Parameters
    ----------
    y : np.ndarray
        y vector of observations
    proba : np.ndarray
        probabilities predicted by the RF model

    Returns
    ------
    LogLoss : float
        The log-loss 
    """

    # since log at 0 is -inf, we add a tiny offset to prevent the equation from blowing up.
    epsilon = 1e-15
    p = np.clip(proba, epsilon, 1-epsilon)

    N = len(y)
    penalty = (y*np.log10(p) + (1-y)*np.log10(1-p))
    LogLoss = -1/N * np.sum(penalty)

    return LogLoss

def calc_bias(proba,y):
    bias = np.mean(proba - y)
    return bias

def calc_timeseries(test_dict,p_dict,calc_metric,times,thresh=None):

    metric_timeseries = np.zeros(len(times))

    for tt in times:
        y_hat = p_dict[f'p{tt}'].flatten()

        # if the threshold is defined, find the deterministic forecast
        if thresh:
            y_hat = np.where(y_hat>=thresh,1,0)

        y_arr = test_dict[f'y{tt}']

        try:
            metric = calc_metric(y_hat,y_arr.flatten())
        except ZeroDivisionError:
            print(tt)
            break

        metric_timeseries[int(tt)] = metric

    return metric_timeseries 

def calc_deterministic_verification(p, y_test):

    threshs = np.linspace(0,1,600) # defines thresholds between 0 and 1 (default is 0.5)

    TPR_arr, FPR_arr, CSI_arr = np.zeros(len(threshs)), np.zeros(len(threshs)), np.zeros(len(threshs))
    
    for i,t in enumerate(threshs): 
        #find where the prediction is greater than or equal to the threshold
        p_bi = np.where(p >= t,1,0)

        try:
            #find statistics for the predictions at this specific threshold
            TPR = stats.calc_TPR(p_bi, y_test)
            FPR = stats.calc_FPR(p_bi, y_test)
            CSI = stats.calc_CSI(p_bi, y_test)
        except ZeroDivisionError:
            break

        TPR_arr[i] = TPR
        FPR_arr[i] = FPR
        CSI_arr[i] = CSI

    # calculate AUC ( were flipping the arrays because they have to be in ascending order for the integration)
    AUC = np.trapz(np.flip(TPR_arr),np.flip(FPR_arr))

    return TPR_arr, FPR_arr, threshs, AUC, CSI_arr

def upsample(X, y, r_f=0.5):
    """
    Upsample the X maxtrix and y vector to acheive a final class ratio given by r_f. 
    To prevent the kernel from crashing, the data gets split into batches then upsampled seperately.
    Oversampling happens first then undersampling second. 

    Parameters
    ----------
    X : np.ndarray
        X matrix of features of shape (n_samples, n_features)
    y : np.ndarray
        y vector of target values of shape (n_samples, )
    r_f: float
        Desired class ratio after upsampling. Value between 0 and 1. If None = 0.5. 

    Returns
    ------
    X_upsampled : np.ndarray
        Upsampeld matrix X 
    y_upsampled : np.ndarray
        Upsampled vector y 

    Notes:
    _____
    This function relies on the SMOTE from the imblearn over_sampling class and RandomeUnderSampler from the under_sampling class 
    """

    n_in_batch = 4432488 # number of samples in one batch (this is the number of samples in 1 days worth of WRF data)

    n_batches = int(X.shape[0] / n_in_batch) # number of batches (number of training days). Its okay if this isnt an int!

    X_batches_lis = np.array_split(X, n_batches) # splitting the arrays doesnt change the order so we can do this to X and y without worrying about the indicies getting mixed up
    y_batches_lis = np.array_split(y, n_batches)

    # empty lists to hold the upsampled data
    X_upsampled_lis = []
    y_upsampled_lis = []

    i=0
    # itterate over each of the batches
    for X_batch,y_batch in zip(X_batches_lis,y_batches_lis):

        # calculate the oversampling class ratio necessary for the final class ratio to equal r_f
        n_obs = len(y_batch)
        y0_i = len(y_batch==0)

        y1_ov = r_f*(n_obs)/(1+r_f)
        r_ov = y1_ov/y0_i

        # initialize oversampling class SMOTE with the calculated ratio
        oversample = imblearn.over_sampling.SMOTE(sampling_strategy=r_ov)
        try:
            X_ov,y_ov = oversample.fit_resample(X=X_batch,y=y_batch)

            # initialize the undersampler with the desired final class ratio 
            undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=r_f)
            X_u,y_u = undersample.fit_resample(X_ov,y_ov)

            # append to the empty lists 
            X_upsampled_lis.append(X_u)
            y_upsampled_lis.append(y_u)
        except ValueError:
            print(f'Batch {i} had no class 1s. Skiped this batch.')
            continue
        i+=1

    # merge the list of arrays together to get the entire upsampled data set
    X_upsampled = np.concatenate(X_upsampled_lis)
    y_upsampled = np.concatenate(y_upsampled_lis)

    return X_upsampled, y_upsampled

def cross_validation(X_train_all,y_train_bi,k):
    sk_folds = StratifiedKFold(n_splits = k, shuffle = False)
    max_samples = int(X_train_all.shape[0]/20)

    # initialize the models 
    RFup = RandomForestClassifier(bootstrap=True, 
                                max_samples=max_samples, 
                                n_estimators=200, 
                                max_depth=None)
    RFw = RandomForestClassifier(bootstrap=True, 
                                max_samples=max_samples, 
                                n_estimators=200, 
                                max_depth=50, 
                                class_weight = 'balanced_subsample')

    # perform the stratified k-fold cross-validation
    validation_dict = {'LLup':[],'LLw':[],'CSIup':[],'CSIw':[],'TPRup':[],'TPRw':[],'FPRup':[],'FPRw':[],'AUCup':[],'AUCw':[],'Bup':[],'Bw':[],'threshs':[]}
    ki = 0
    for train_ii, test_ii in sk_folds.split(X_train_all, y_train_bi):
        print(ki)
        ki +=1

        # index all the data according to the split indices
        X_train = X_train_all[train_ii]
        X_test = X_train_all[test_ii]
        y_train = y_train_bi[train_ii]
        y_test = y_train_bi[test_ii]
        print('index done')

        # make an upsampled version of the training data
        X_train_up, y_train_up = stats.upsample(X_train,y_train)
        print('upsample done')

        # fit the models
        rf_up_fit = RFup.fit(X_train_up,y_train_up) # RF with upsampled data
        rf_w_fit = RFw.fit(X_train,y_train)
        print('done fitting')

        # obtain the lightning probabilities (test on the SAME data)
        pup = rf_up_fit.predict_proba(X_test)[:,1]
        pw = rf_w_fit.predict_proba(X_test)[:,1]

        # append to lists in the dictionary 
        case_names = ['up','w']
        probabilities = [pup,pw]
        for name,prob in zip(case_names,probabilities): 

            # calculate the deterministic verification statistics
            TPR, FPR, threshs, AUC, CSI = stats.calc_deterministic_verification(prob, y_test)

            # calculate the probablistic verification statistics 
            LL = stats.calc_LogLoss(prob, y_test)
            B = stats.calc_bias(prob, y_test)

            # append to lists in the dictionary 
            validation_dict[f'LL{name}'].append(LL)
            validation_dict[f'B{name}'].append(B)
            validation_dict[f'TPR{name}'].append(TPR)
            validation_dict[f'FPR{name}'].append(FPR)
            validation_dict[f'CSI{name}'].append(CSI)
            validation_dict[f'AUC{name}'].append(AUC)

    # thresholds dont change between itterations
    validation_dict[f'threshs'].append(threshs)

    # take the average of the three folds (some variables require the 'axis' argument because they are arrays)
    for name in case_names:
        validation_dict[f'LL{name}'] = np.mean(validation_dict[f'LL{name}'])
        validation_dict[f'B{name}'] = np.mean(validation_dict[f'B{name}'])
        validation_dict[f'TPR{name}'] = np.mean(validation_dict[f'TPR{name}'],axis=0)
        validation_dict[f'FPR{name}'] = np.mean(validation_dict[f'FPR{name}'],axis=0)
        validation_dict[f'CSI{name}'] = np.mean(validation_dict[f'CSI{name}'],axis=0)
        validation_dict[f'AUC{name}'] = np.mean(validation_dict[f'AUC{name}'])

    # save the dictionary
    with open('validation_dict.pkl','wb') as f:
        pickle.dump(validation_dict,f)

    return

def train_preprocess(training_dates,times):
    for date in training_dates:
        s = time()
        name_train = 'wrfout_d02_'+date[0]+date[1]+date[2]+'_merged.nc'
        y_train, X_train = get_X_y(name_train,date[0],date[1],date[2],times)
        np.save('preprocessed/y_train/y_train'+date[0]+date[1]+date[2]+'.npy',y_train)
        np.save('preprocessed/X_train/X_train'+date[0]+date[1]+date[2]+'.npy',X_train)
        e = time()
        print(f'time for one day: {e-s}')
    return

def test_preprocess(testing_dates,times):
    for date in testing_dates:
        for t in times:
            try:
                name_test = 'wrfout_d02_'+date[0]+'-'+date[1]+'-'+date[2]+'_'+t+'_00_00'
                y_test, X_test = get_X_y(name_test,date[0],date[1],date[2],[t])
                np.save('preprocessed/feature_subset/y_test/y_test'+date[0]+date[1]+date[2]+t+'.npy',y_test)
                np.save('preprocessed/feature_subset/X_test/X_test'+date[0]+date[1]+date[2]+t+'.npy',X_test)
            except FileNotFoundError:
                break
    return

def retreive_X_y_train(dates,feature_lis,shape):
    times = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    y_lis = []
    X_lis = []
    for date in dates:
        y = np.load('preprocessed/full_domain/y_train/y_train'+date[0]+date[1]+date[2]+'.npy')
        X = np.load('preprocessed/full_domain/X_train/X_train'+date[0]+date[1]+date[2]+'.npy')
        y_sub, X_sub = subset_X_y(X,y,date,times,feature_lis,shape)
        y_lis.append(y_sub)
        X_lis.append(X_sub)
    
    y = np.concatenate(y_lis)
    X = np.concatenate(X_lis)

    return y, X

def retreive_X_y_train_small(dates,feature_lis,shape):
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

def retreive_X_y_test(date,time,feature_lis,shape):
    """""
    Note that if you want to run the model on the full domain, you should comment out the subet_X_y function.
    """""
    times = [time]
    y_full = np.load('preprocessed/feature_subset/y_test/y_test'+date[0]+date[1]+date[2]+times[0]+'.npy')
    X_full = np.load('preprocessed/feature_subset/X_test/X_test'+date[0]+date[1]+date[2]+times[0]+'.npy')

    y, X = subset_X_y(X_full,y_full,date,times,feature_lis,shape)
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

def reconstruct_domain(var,shape):
    full_arr = np.zeros(shape=shape)

    with open('coord_dict.pkl', 'rb') as f:
        coord_dict = pickle.load(f)
    
    i = 0
    for coord, val in coord_dict.items():
        if val != -999:
            full_arr[coord] = var[i]
            i+=1
        else:
            full_arr[coord] = np.nan
    
    return full_arr

