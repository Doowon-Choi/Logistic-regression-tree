# MOB Logistic Regression Tree

The Model-based Recursive partitioning for binary outcome (i.e., MOB logistic regression tree) is implemented in Python from scratch.   
The MOB tree is already built in R package *"party"* and *"partykit"*, and it is able to handle a variety classes of parametric models like linear regression, 
logistic regression and so on. Details are found in the reference below.

## Train MOB tree
Before fitting MOB logistic regression tree, the variables in the dataset should be carefully designed.   

The first column of dataset must be a response variable (i.e., binary outcome) having 0 or 1.
Predictor variables X (i.e., variables used to fit logistic regression) have to be placed from the second column. For example, if we have three predictors, those three are placed
in the second, third, and fourth column of dataset. The rest columns are seet to split variables Z (i.e., variables for split a node).   

Note that the idea of MOB tree is that if there is a sytematic change in parameters of models (i.e., coefficients) with respect to split variable at certain node, the node is split
by split variable that leads to the biggest change.   

For a categorical variable in dataset, please define it as 'object' type like the following way.   
```python
df["gender"] = df["gender"].astype('object')
```
The following function LR_createTree is used to build MOB tree.   
```python
model3_2 = LR_createTree(df, df2, [1], [0,1], 6, LR_Accuracy, LR_Accuracy, tolN = 40)
```   
In the above, 'df' is the dataframe of the original dataset (i.e., imported data by pandas), df2 is the numpy array of the dataframe.
   
The third argument is the index of predictor/covariate, and the fourth argument is the index of paramters of our interest. In the above example,
a single predictor is used for building a logistic regression, and tree pays attention to the change of both intercept (index of 0) and slope (index of 1).   

If the third argument is set to [1,2,3] and the fourth argument is set to [1,3], three predictors (i.e., the second, third and fourth column in the dataset) are used
for building a logistic regression, and tree pays attention to the changes in parameters that correspond to the first and third predictor with respect to split variables.

To control tree size, the fifth argument and the last argument, tolN, are used. The fifth argument controls the number of executing 'LR_createTree' function
(i.e., how many splits are tried) and tolN is the minimum sample size in each node. The larger the fifth argument is and the smaller tolN is, the larger/deeper tree is.  

Basically, this version relies on pre-pruning by following the original MOB. However, as the original paper mentions, we can conduct post-pruning via 
large significant level in parameter instablity test and BIC score over large dataset.

For conducting parameter instablity test, 'p_val_set.csv' file is required. The file is used to compute p-value given the limiting distribution *k*-dimensional tied-down Bessel process for continuous split variable. I intentionally did not upload the file here due to potential infringement of intellectual property rights, but users can find it from the original R package "*party*" or Dr. Bruce E. Hansen's work.

This implemtnation use Python dictionary to store the resulting tree.

## Compare results with R
This code reproduces the MOB tree described in the original paper. The dataset is Pima Indians Diabetes data in R pacakge "mlbench", and the modified data is also uploaded.  

![Compare](https://user-images.githubusercontent.com/69023373/89147894-bd9a5f00-d51d-11ea-8743-c894f7100a19.png)

For R code to build MOB tree, please follow the code that is already provided in R package "*Party*".

```r   
library(party)
data("PimaIndiansDiabetes", package = "mlbench")
## partition logistic regression diabetes ~ glucose 
## wth respect to all remaining variables
fmPID <- mob(diabetes ~ glucose | pregnant + pressure + triceps + 
               insulin + mass + pedigree + age,
             data = PimaIndiansDiabetes, model = glinearModel, 
             family = binomial())
fmPID
plot(fmPID)
```
Please take a look at MOB_Main.py file.

## Version
- python 3.6.3
- numpy 1.16.4
- scipy 1.3.1
- pandas 0.20.3
- sklearn 0.21.1

## Reference
Zeileis, A., Hothorn, T., & Hornik, K. (2008). Model-based recursive partitioning. *Journal of Computational and Graphical Statistics*, 17(2), 492-514.   
Hansen, B. E. (1997). Approximate asymptotic p values for structuras-change tests. *Journal of Business & Economic Statistics*, 15(1), 60-67.
