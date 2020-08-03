# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:18:32 2020

@author: Doowon
"""
import numpy as np
import scipy.stats as sttool
import math as mt
import pandas as pd
import itertools as itts
import numpy.linalg as npl
import scipy as sp
from sklearn.linear_model import LogisticRegression

def Standard_LR(dataSet, covariate_ind):
    X = dataSet[:, covariate_ind]
    y = dataSet[:,0]
    n_obs = np.shape(dataSet)[0]
         
    clf = LogisticRegression(fit_intercept=True, C = 1e15,solver='lbfgs')
    clf.fit(X, y)
    interce = clf.intercept_[0]
    coefs = clf.coef_[0]
    b=np.append(interce,coefs)
    weights = np.ones(n_obs)[:,None]
    return [b,weights]

def LR_est(dataSet, covariate_ind):
    X = dataSet[:, covariate_ind]
    y = dataSet[:,0]
    
    if (y.sum()==0):
        return np.append(np.array([-np.inf]),np.zeros(np.shape(covariate_ind)[0]))
        
    clf = LogisticRegression(fit_intercept=True, C = 1e15, solver='lbfgs')
    clf.fit(X, y)
    interce = clf.intercept_[0]
    coefs = clf.coef_[0]
    b=np.append(interce,coefs)
    return b

def sorted_k_partitions(seq, k):

    n = len(seq)
    groups = []  # a list of lists, currently empty

    def generate_partitions(i):
        if i >= n:
            yield list(map(tuple, groups))
        else:
            if n - i > k - len(groups):
                for group in groups:
                    group.append(seq[i])
                    yield from generate_partitions(i + 1)
                    group.pop()

            if len(groups) < k:
                groups.append([seq[i]])
                yield from generate_partitions(i + 1)
                groups.pop()

    result = generate_partitions(0)

    result = [sorted(ps, key = lambda p: (len(p), p)) for ps in result]
    result = sorted(result, key = lambda ps: (*map(len, ps), ps))

    return result
    
def binSplitDataSet(dataSet, feature, value):
    # the first [0] represents a row index meeting the condition
    right_node = np.nonzero(dataSet[:,feature]>value)[0]
    left_node = np.nonzero(dataSet[:,feature]<=value)[0]
    if (len(right_node)<10) or (len(left_node)<10):
        return [], []
    mat0 = dataSet[left_node,:]
    mat1 = dataSet[right_node,:]
    return mat0, mat1
    
def cate_binSplitDataSet(df_original, dataSet, featIndex, splitVal1):
    df1 = df_original
    feat_name = df1.columns[featIndex]
    # convert array into data frame
    df = pd.DataFrame(dataSet, index=dataSet[:,0], columns=df1.columns)
    left_node = df.loc[df[feat_name].isin(list(splitVal1))]
    right_node = df.loc[~df[feat_name].isin(list(splitVal1))]
    if (len(right_node)<10) or (len(left_node)<10):
        return [], []
    mat0 = left_node.as_matrix()
    mat1 = right_node.as_matrix()
    return mat0, mat1

    
########## find estimated beta and weights #############

##### log-likelihood function ####
def LR_log_likelihood(dataSet, covariate_ind): # error term, ingredient of split criterion
    if (len(dataSet)==0):
        return 0
    eps = 1e-8
    mod1 = Standard_LR(dataSet, covariate_ind)
    est_betas = mod1[0].astype(np.float32)

    # calculate estimated probability and corresponding class
    X = dataSet[:,covariate_ind]#[:,None]  ### 398*1 dimension (398,) --> (398,1)
    y = dataSet[:,0]#[:,None]
    n_obs = np.shape(dataSet)[0]
  
    # design matrix
    intercept = np.ones(n_obs)[:,None]
    X1 = np.hstack((intercept, X))
     
    score = np.dot(X1,est_betas).astype(np.float32)
    mu=1/(1+np.exp(-score))
      
    ## calculate deviances ## for maximize -deviance
    deviances = -2*(y*np.log(mu+eps)+(1-y)*np.log(1-mu+eps)).astype(np.float32)
    ll = np.sum(-deviances)
       
    return ll
    
def LR_Find_split_var_by_MOB(df_original, dataSet, covariate_ind, param_interest_index):
    df = df_original; dataSet=dataSet;
    param_interest_index = param_interest_index
    covariate_ind = covariate_ind
    res1 =  Standard_LR(dataSet, covariate_ind)
    betas = res1[0].astype(np.float32); h_weights = res1[1].astype(np.float32)
    k = betas.shape[0]
    del res1
    
    num_obs, num_feature = np.shape(dataSet)

    #### ingredient for calculating score ###
    X = dataSet[:,covariate_ind]
    y = dataSet[:,0]
    n_obs = np.shape(dataSet)[0]
  
    # design matrix
    intercept = np.ones(n_obs)[:,None]
    X1 = np.hstack((intercept, X)) 
    
    ##calculate score ##
    mu = 1/(1+np.exp(-np.dot(X1,betas)))
    resid = (y-mu)[:,None] 
    res_mat =np.tile(resid,(1,k))
    weights_mat = np.tile(h_weights,(1,k))
    score = weights_mat*X1*res_mat
    
    ### Covariance matrix of the score function using outer product of the score
    process = score/np.sqrt(n_obs)
    ## find root matrix
    J12=sp.linalg.sqrtm(np.dot(process.T,process))
    ## cholesky decomposition
    cho_j12=sp.linalg.cholesky(J12, lower=False)
    Q, R = sp.linalg.qr(cho_j12)
    cov_mat = sp.linalg.inv(np.dot(R.T,R))
    c_process = np.dot(cov_mat,process.T).T
       
    c_process = c_process[:,param_interest_index]
    k = len(param_interest_index)
    
    pval_list=[]
    start_feature = 1 + np.shape(covariate_ind)[0]
    for featIndex in range(start_feature,num_feature):
        epsilon = 1e-8 ###########################################
        if df_original[df_original.columns[featIndex]].dtypes=='object': ### categorical split variable
            zi = dataSet[:,featIndex]
            oi = np.argsort(zi)
            zi = zi[oi][:,None]
                 
            proci = c_process[oi]
            unique, counts = np.unique(zi, return_counts=True)
            segweights = counts/n_obs
            segweights = segweights[:,None]
            
            #aggregate proci and zi
            dt_set = np.hstack((proci,zi))
            col_name = df_original.columns[1:start_feature].tolist()
            col_name.insert(0, 'b0') #put 'b0' at the first position
            
            ### select column name corresponding parameter of our interest
            fin_col_name = [col_name[index] for index in param_interest_index]
            
            ### calculate test statsitic
            fin_col_name.append('id')
            tmp_df=pd.DataFrame(dt_set, columns=fin_col_name)
            a2=tmp_df.groupby('id').sum()
            t1=np.power(a2,2)/np.tile(segweights,(1,k))
           
            stat = t1.sum().sum()
            pval = (1 - sttool.chi2.cdf(stat,k*(len(unique)-1)))
            sub_dta=[featIndex,df_original.columns[featIndex],stat,pval]
            pval_list.append(sub_dta)
            
        else: ### continuous split variable
            zi = dataSet[:,featIndex]
            oi = np.argsort(zi)
            zi = zi[oi][:,None]
            n = n_obs
            proci = c_process[oi]

            #home
            b_set=pd.read_csv("C:/Users/Doowon/Documents/Python_DT/p_val_set.csv")
            beta= b_set.as_matrix()[:,[1,2,3]]
            trim = 0.1
            minsplit = 20
            from1 = np.ceil(n*trim)
            from1 = max(from1,minsplit)
            to1 = n - from1
            lambda1 = ((n-from1)*to1)/(from1*(n-to1))
            
            proci1 = np.cumsum(proci,axis=0)
            from1 = int(from1)
            to1 = int(to1)
            xx = np.sum(proci1**2,axis=1)[:,None] ### row sum
            xx = xx[from1:to1+1]        
            tt =  np.arange(from1, to1+1)[:,None]/n
            stat_set = np.divide(xx,tt*(1-tt))
            stat_set =xx/(tt*(1-tt))
            stat = np.amax(stat_set) ### test statistics
            
            #### calculate p-value ####
            m = np.shape(beta)[1]-1
            if (lambda1 < 1):
                tau = lambda1
            else: 
                tau = 1/(1+np.sqrt(lambda1))
            beta = beta[np.arange((k-1)*25,k*25),:]
            f_term = beta[:,0:m]
            s_term = stat**np.arange(0,m)
            dummy = np.sum(f_term*s_term,axis=1)
            dummy = dummy*(dummy>0)
            pp = np.log((1 - sttool.chi2.cdf(dummy,beta[:,m]))+epsilon) 
            
            if (tau==0.5):
                p = np.log((1 - sttool.chi2.cdf(stat,k)))
            elif (tau <= 0.01):
                p = pp[25]
            elif (tau >=0.49):
                p = np.log((np.exp(np.log(0.5-tau)+pp[0])+np.exp(np.log(tau-0.49)+\
                           np.log((1 - sttool.chi2.cdf(stat,k))+epsilon)))*100)
                
            else:
                taua = (0.51-tau)*50
                tau1 = np.floor(taua)
                tau1 = int(tau1)
                p = np.log(np.exp(np.log(tau1+1-taua)+pp[tau1-1])+np.exp(np.log(taua-\
                           tau1)+pp[tau1]))
            ### find p-value ####
            pval=np.exp(p)
            sub_dta=[featIndex,df_original.columns[featIndex],stat,pval]
            pval_list.append(sub_dta)
            
    set1 = pd.DataFrame(pval_list,  columns=['index','variable','stat','pval'])
    ### p-value = 1 means that that variable is used split varialbe before
    tps1 = len(set1[(set1['pval']==1)])
    
    ### corrected p-value using Sidak's method ###
    set1['pval']=1-(1-set1['pval'])**(num_feature-2-tps1)
    min_dta = set1[set1['pval']==set1['pval'].min()]
    if (np.shape(min_dta)[0]==0):
        return None

    if (np.shape(min_dta)[0]!=1):
        bestIndex=min_dta[min_dta['stat']==min_dta['stat'].max()]['index'].values[0]
    else:
        bestIndex = min_dta['index'].values[0]
    
    ### be careful when panda is used as argument in if statment
    if ((min_dta['pval']<0.05).all()):
        return bestIndex
    else:
        return None
        
##############################################################################
        
def LR_pred_rate(dataSet, covariate_ind): # error term, ingredient of split criterion
    if (len(dataSet)==0):
        return 0
    mod1 = Standard_LR(dataSet, covariate_ind)
    est_betas = mod1[0].astype(np.float32)

    # calculate estimated probability and corresponding class
    X = dataSet[:,covariate_ind]#[:,None]  ### 398*1 dimension (398,) --> (398,1)
    y = dataSet[:,0][:,None]
    n_obs = np.shape(dataSet)[0]
  
    # design matrix
    intercept = np.ones(n_obs)[:,None]
    X1 = np.hstack((intercept, X))
    mu = 1/(1+np.exp(-np.dot(X1,est_betas)))
    pred = np.where(mu >= 0.5, 1, 0)[:,None]
  
    temp1 = np.hstack((y,pred))

    ### generate contigency table for comparision between actual and pred
    c_table = pd.DataFrame(temp1, columns=['actual','pred'])
    tn = np.sum((pred==0) & (y==0))
    tp = np.sum((pred==1) & (y==1))

    return (tn+tp)/n_obs

def LR_pred_residual(dataSet, covariate_ind): # error term, ingredient of split criterion
    if (len(dataSet)==0):
        return 0
    mod1 =  Standard_LR(dataSet, covariate_ind)
    est_betas = mod1[0].astype(np.float32)
        
    # calculate estimated probability and corresponding class
    X = dataSet[:,covariate_ind]#[:,None]  ### 398*1 dimension (398,) --> (398,1)
    y = dataSet[:,0]#[:,None]
    n_obs = np.shape(dataSet)[0]
  
    # design matrix
    intercept = np.ones(n_obs)[:,None]
    X1 = np.hstack((intercept, X))
    mu = 1/(1+np.exp(-np.dot(X1,est_betas)))
    
    temp2 = np.abs(y-mu)
    sum_abs_residual = np.sum(temp2)
  
    return sum_abs_residual

## calculate misclassification rate
def LR_Accuracy(dataSet, covariate_ind): # error term, ingredient of split criterion
    if (len(dataSet)==0):
        return 0
    mod1 =  Standard_LR(dataSet, covariate_ind)
    est_betas = mod1[0].astype(np.float32)
   
    # calculate estimated probability and corresponding class
    X = dataSet[:,covariate_ind] 
    y = dataSet[:,0][:,None]
    n_obs = np.shape(dataSet)[0]
  
    # design matrix
    intercept = np.ones(n_obs)[:,None]
    X1 = np.hstack((intercept, X))
    mu = 1/(1+np.exp(-np.dot(X1,est_betas)))
    pred = np.where(mu >= 0.5, 1, 0)[:,None]
    
    temp1 = np.hstack((y,pred))

    ### generate contigency table for comparision between actual and pred
    c_table = pd.DataFrame(temp1, columns=['actual','pred'])
    tn = np.sum((pred==0) & (y==0))
    tp = np.sum((pred==1) & (y==1))
        
    #return conf_mat
    return tn, tp, n_obs

def LR_createTree(df_original, dataSet, covariate_ind, param_interest_index,  max_num_create,
                  leafType = LR_est, errType = LR_Accuracy, tolN = 200):
    
    global  num_create
    param_interest_index = param_interest_index
    tolN  = tolN 
    cov_ind = covariate_ind
    num_create =  num_create + 1
    
    if  num_create > max_num_create: return leafType(dataSet, cov_ind)
    
    ############# do not split subgroups where all values of y are 0
    if (dataSet[:,0].sum()==0): return leafType(dataSet, cov_ind)   
    
    df_original = df_original
    feat, val, val_sp = LR_chooseBestSplit(df_original, dataSet, cov_ind, param_interest_index, leafType, errType, tolN) 
    
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = df_original.columns[feat] 
    retTree['spVal'] = val_sp ## this used for finding split variables
    
    if df_original[df_original.columns[feat]].dtypes=='object':
        lSet, rSet = cate_binSplitDataSet(df_original,dataSet,feat,val_sp)
    else:
        lSet, rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left']=LR_createTree(df_original, lSet, cov_ind, param_interest_index, max_num_create, leafType, errType, tolN)
    retTree['right']=LR_createTree(df_original,rSet, cov_ind, param_interest_index, max_num_create, leafType, errType, tolN)
    
    return retTree
    
###### NOTE THAT tolS is a fixed value, how much do two subgroup are different in cor
def LR_chooseBestSplit(df_original, dataSet, covariate_ind, param_interest_index,leafType=LR_est, errType = LR_Accuracy, tolN = 200):
    param_interest_index = param_interest_index
    df_original = df_original; tolN = tolN
    m,n = np.shape(dataSet)
    cov_ind = covariate_ind  
    
    #####  objective function ##
    S  = LR_log_likelihood(dataSet, cov_ind)
    bestS = -np.inf; bestIndex = 0; bestValue_for_split=0;bestValue_for_return = 0;
    
    featIndex = LR_Find_split_var_by_MOB(df_original, dataSet, cov_ind, param_interest_index=param_interest_index)
    if featIndex == None: return None, leafType(dataSet, cov_ind), None
    
    if df_original[df_original.columns[featIndex]].dtypes=='object':
       categories = list(set(dataSet[:,featIndex].flat))
       poss_comb = sorted_k_partitions(categories,2)
            
       for splitVal in poss_comb:
           splitVal1 = splitVal[0]
           mat0, mat1 = cate_binSplitDataSet(df_original,dataSet, featIndex, splitVal1)
           if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] <tolN):
               continue
           if (mat0[:,0].sum()==0):
               log_left = 0
           else:
               #print(splitVal)
               log_left = LR_log_likelihood(mat0, cov_ind)
               
           if (mat1[:,0].sum()==0):
               log_right = 0
           else:
               #print(splitVal)
               log_right = LR_log_likelihood(mat1, cov_ind)
           ## if log-likelihood 
           newS = log_left + log_right
           if newS > bestS : #eqaul and greater to
              bestIndex = featIndex
              bestValue_for_split = splitVal1
              bestValue_for_return = splitVal1[0]
              bestS = newS

    else:        
       for splitVal in set(dataSet[:,featIndex].flat):#give value of the feature
           mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
           if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] <tolN):
               continue
       
           if (mat0[:,0].sum()==0):
               log_left = 0
           else:
               #print(splitVal)
               log_left = LR_log_likelihood(mat0, cov_ind)
               
           if (mat1[:,0].sum()==0):
               log_right = 0
           else:
               #print(splitVal)
               log_right = LR_log_likelihood(mat1, cov_ind)
           ## if log-likelihood 
           newS = log_left + log_right
           #newS = LR_log_likelihood(mat0, cov_ind) + LR_log_likelihood(mat1, cov_ind)
           if newS > bestS: #since maximize newS
              bestIndex = featIndex
              bestValue_for_split = splitVal
              bestValue_for_return = splitVal
              bestS = newS
     
    if (bestS<S):
        return None, leafType(dataSet, cov_ind), None
        
    if df_original[df_original.columns[bestIndex]].dtypes=='object':
        mat0, mat1 = cate_binSplitDataSet(df_original,dataSet, bestIndex, bestValue_for_split)
    else:
        mat0, mat1=binSplitDataSet(dataSet, bestIndex, bestValue_for_split)
    
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] <tolN) :
        return None,leafType(dataSet, cov_ind), None
        
    return bestIndex, bestValue_for_return, bestValue_for_split
    
def split_info(df_original, dataSet, covariate_ind, param_interest_index, selected_split_variable):
    df = df_original; dataSet=dataSet;
    param_interest_index = param_interest_index
    covariate_ind = covariate_ind
    res1 =  Standard_LR(dataSet, covariate_ind)
    betas = res1[0].astype(np.float32); h_weights = res1[1].astype(np.float32)
    k = betas.shape[0]
    del res1
    
    num_obs, num_feature = np.shape(dataSet)

    #### ingredient for calculating score ###
    X = dataSet[:,covariate_ind]
    y = dataSet[:,0]
    n_obs = np.shape(dataSet)[0]
  
    # design matrix
    intercept = np.ones(n_obs)[:,None]
    X1 = np.hstack((intercept, X)) 
    
    ##calculate score ##
    mu = 1/(1+np.exp(-np.dot(X1,betas)))
    resid = (y-mu)[:,None] 
    res_mat =np.tile(resid,(1,k))
    weights_mat = np.tile(h_weights,(1,k))
    score = weights_mat*X1*res_mat
    
    ### Covariance matrix of the score function using outer product of the score
    process = score/np.sqrt(n_obs)
    ## find root matrix
    J12=sp.linalg.sqrtm(np.dot(process.T,process))
    ## cholesky decomposition
    cho_j12=sp.linalg.cholesky(J12, lower=False)
    Q, R = sp.linalg.qr(cho_j12)
    cov_mat = sp.linalg.inv(np.dot(R.T,R))
    c_process = np.dot(cov_mat,process.T).T
       
    c_process = c_process[:,param_interest_index]
    k = len(param_interest_index)
    
    ## obtain column index from column name in dataframe
    featIndex = df.columns.get_loc(selected_split_variable)
    start_feature = 1 + np.shape(covariate_ind)[0]
    epsilon = 1e-8 
    
    if df_original[df_original.columns[featIndex]].dtypes=='object':
        ### categorical split variable
        zi = dataSet[:,featIndex]
        oi = np.argsort(zi)
        zi = zi[oi][:,None]
             
        proci = c_process[oi]
        unique, counts = np.unique(zi, return_counts=True)
        segweights = counts/n_obs
        segweights = segweights[:,None]
        
        #aggregate proci and zi
        dt_set = np.hstack((proci,zi))
        col_name = df_original.columns[1:start_feature].tolist()
        col_name.insert(0, 'b0') #put 'b0' at the first position
        
        ### select column name corresponding parameter of our interest
        fin_col_name = [col_name[index] for index in param_interest_index]
        
        ### calculate test statsitic
        fin_col_name.append('id')
        tmp_df=pd.DataFrame(dt_set, columns=fin_col_name)
        a2=tmp_df.groupby('id').sum()
        t1=np.power(a2,2)/np.tile(segweights,(1,k))
       
        stat = t1.sum().sum()
        pval = (1 - sttool.chi2.cdf(stat,k*(len(unique)-1)))
        
        return stat, pval
        
    else: ### continuous split variable
        zi = dataSet[:,featIndex]
        oi = np.argsort(zi)
        zi = zi[oi][:,None]
        n = n_obs
        proci = c_process[oi]

        #home
        b_set=pd.read_csv("C:/Users/Doowon/Documents/Python_DT/p_val_set.csv")
        beta= b_set.as_matrix()[:,[1,2,3]]
        trim = 0.1
        minsplit = 20
        from1 = np.ceil(n*trim)
        from1 = max(from1,minsplit)
        to1 = n - from1
        lambda1 = ((n-from1)*to1)/(from1*(n-to1))
        
        proci1 = np.cumsum(proci,axis=0)
        from1 = int(from1)
        to1 = int(to1)
        xx = np.sum(proci1**2,axis=1)[:,None] ### row sum
        xx = xx[from1:to1+1]        
        tt =  np.arange(from1, to1+1)[:,None]/n
        stat_set = np.divide(xx,tt*(1-tt))
        stat_set =xx/(tt*(1-tt))
        stat = np.amax(stat_set) ### test statistics
        
        #### calculate p-value ####
        m = np.shape(beta)[1]-1
        if (lambda1 < 1):
            tau = lambda1
        else: 
            tau = 1/(1+np.sqrt(lambda1))
        beta = beta[np.arange((k-1)*25,k*25),:]
        f_term = beta[:,0:m]
        s_term = stat**np.arange(0,m)
        dummy = np.sum(f_term*s_term,axis=1)
        dummy = dummy*(dummy>0)
        
        pp = np.log((1 - sttool.chi2.cdf(dummy,beta[:,m]))+epsilon) 
        
        if (tau==0.5):
            p = np.log((1 - sttool.chi2.cdf(stat,k)))
        elif (tau <= 0.01):
            p = pp[25]
        elif (tau >=0.49):
            p = np.log((np.exp(np.log(0.5-tau)+pp[0])+np.exp(np.log(tau-0.49)+\
                       np.log((1 - sttool.chi2.cdf(stat,k))+epsilon)))*100)
            
        else:
            taua = (0.51-tau)*50
            tau1 = np.floor(taua)
            tau1 = int(tau1)
            p = np.log(np.exp(np.log(tau1+1-taua)+pp[tau1-1])+np.exp(np.log(taua-\
                       tau1)+pp[tau1]))
        ### find p-value ####
        pval=np.exp(p)
        
        return stat, pval

    
 ###########################################################################
