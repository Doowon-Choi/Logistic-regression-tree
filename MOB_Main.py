# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as sttool
import math as mt
import pandas as pd
import itertools as itts
import numpy.linalg as npl
import scipy as sp
from sklearn.linear_model import LogisticRegression

####
df=pd.read_csv("C:/Users/Doowon/Documents/Python_DT/diabete_reformed.csv")
df1=df.dropna()# remove na missing records
df2 = df.as_matrix()
num_create = 0

model3_2 = LR_createTree(df, df2, [1], [0,1], 6, LR_Accuracy, LR_Accuracy, tolN = 40)


### Find subgroup-specific logitistic regression
node2 = df.loc[df.mass<= 26.3,:]  
node2_mat = node2.as_matrix()
Standard_LR(node2_mat, [1])[0] #array([-9.9515095 ,  0.05870786])

node3 = df.loc[df.mass > 26.3,:]  
node3_mat = node3.as_matrix()

node4 = df.loc[(df.mass > 26.3) & (df.age <= 30),:]  
node4_mat = node4.as_matrix()
Standard_LR(node4_mat, [1])[0] #array([-6.70558567,  0.04683748])


node5 = df.loc[(df.mass > 26.3) & (df.age > 30),:]  
node5_mat = node5.as_matrix()
Standard_LR(node5_mat, [1])[0] # array([-2.77095416,  0.02353582])

### Split information ####
## node 1 = the entire dataset
split_info(df, df2, [1], [0,1], "mass")# 47.786694163594426, 1.198716031419208e-08

split_info(node3, node3_mat,[1], [0,1], "age")# 34.55681194571885, 1.4373662459909676e-06