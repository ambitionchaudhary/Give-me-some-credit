# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 04:59:23 2018

@author: Ambition
"""

#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (12.0, 10.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import graphviz
from sklearn import preprocessing,model_selection
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import xgboost as xgb

#loading data
train_df = pd.read_csv("C:/Users/Ambition/Documents/data analytics/Give me some credit/cs-training.csv")
test_df = pd.read_csv("C:/Users/Ambition/Documents/data analytics/Give me some credit/cs-test.csv")

train.head()
train.info()
train.describe()
train2=train.copy()
test2=test.copy()

##assign new column ‘id’ to test and training data set
col_names = train2.columns.values
col_names[0] = 'ID' 
train2.columns = col_names 
test2.columns = col_names 

#Data exploration and visualization
train2['SeriousDlqin2yrs'].value_counts()
sns.countplot(x='SeriousDlqin2yrs', data=train2, palette='hls')
plt.show()
plt.savefig('count_plot')
train2.groupby('SeriousDlqin2yrs').mean()


train2.age.hist()  ##age histogram 
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')

train.columns[train.isnull().any()]  ##check missing values


miss = train.isnull().sum()/len(train) ##missing value counts in each of these column
miss = miss[miss > 0]
miss.sort_values(inplace=True)
miss

miss = miss.to_frame()  ##visualising missing values

miss.columns = ['count']
miss.index.names = ['Features']
miss['Features'] = miss.index

sns.set(style="whitegrid", color_codes=True)  ##plot the missing value count

sns.barplot(x = 'Features', y = 'count', data=miss)
plt.xticks(rotation = 90)
sns.plt.show()

#finding skewness
feature_list=list(train2.columns.values)
remove_list = ['ID','SeriousDlqin2yrs','MonthlyIncome','NumberOfDependents']   ## remove ID, target variable Dlqin2yrs and variables with missing values

for each in remove_list:
    feature_list.remove(each)

for each in feature_list:    
    sns.distplot(train2[each])
    plt.show()

x=train2['SeriousDlqin2yrs']     ## Plot RevolvingUtilizationOfUnsecuredLines (%<1)
y=train2['RevolvingUtilizationOfUnsecuredLines']
colors = (0,0,0)
area = np.pi*3
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('RevolvingUtilizationOfUnsecuredLines outliers')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x=train2['SeriousDlqin2yrs']     ## Plot DebtRatio (should be between 0 and 1)
y=train2['DebtRatio']
colors = (0,0,0)
area = np.pi*3
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('DebtRatio outliers')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

train2['NumberOfTime30-59DaysPastDueNotWorse'].value_counts()   ##number of times each value appears
train2['NumberOfTimes90DaysLate'].value_counts()
train2['NumberOfTime60-89DaysPastDueNotWorse'].value_counts()



#Data preprocessing

train2.drop(train2[(train2['age']==0)].index, inplace=True)

train2['MonthlyIncome'].median()     ##filling missing values
train2['MonthlyIncome'].fillna((train2['MonthlyIncome'].median()), inplace=True)

train2['NumberOfDependents'].median()
train2['NumberOfDependents'].fillna((train2['NumberOfDependents'].median()), inplace=True)

logtrans= train2.columns.values[[2,4,5,8,9,10]]  ##log transformation
logtrans
for each in logtrans:
    train2[each] = np.log(1+train2[each].values)
    
    for each in feature_list:
    sns.distplot(train2[each])
    plt.show()
    
    train2['MonthlyIncome'] = np.log(1+train2['MonthlyIncome'].values)
train2['NumberOfDependents'] = np.log(1+train2['NumberOfDependents'].values)
sns.distplot(train2['MonthlyIncome'])
plt.show()
sns.distplot(train2['NumberOfDependents'])
plt.show()

train2.drop(train2[(train2['DebtRatio']>=2) & (train2['DebtRatio']==0)].index, inplace=True) ##drop missing values from debt ratio since values are too high and have many missing values in the row==0

median = train2.loc[train2['NumberOfTime30-59DaysPastDueNotWorse']<10, 'NumberOfTime30-59DaysPastDueNotWorse'].median()
train2.loc[train2['NumberOfTime30-59DaysPastDueNotWorse'] > 10, 'NumberOfTime30-59DaysPastDueNotWorse'] = np.nan
train2['NumberOfTime30-59DaysPastDueNotWorse'].fillna(median,inplace=True)

median1 = train2.loc[train2['NumberOfTimes90DaysLate']<20, 'NumberOfTimes90DaysLate'].median()
train2.loc[train2['NumberOfTimes90DaysLate'] > 20, 'NumberOfTimes90DaysLate'] = np.nan
train2['NumberOfTimes90DaysLate'].fillna(median,inplace=True)

median2 = train2.loc[train2['NumberOfTime60-89DaysPastDueNotWorse']<20, 'NumberOfTime60-89DaysPastDueNotWorse'].median()
train2.loc[train2['NumberOfTime60-89DaysPastDueNotWorse'] > 20, 'NumberOfTime60-89DaysPastDueNotWorse'] = np.nan
train2['NumberOfTime60-89DaysPastDueNotWorse'].fillna(median,inplace=True)

#replacing values of RevolvingUtilizationOfUnsecuredLines greater than 1 with median and drop==0 since ratio
train2.drop(train2[(train2['RevolvingUtilizationOfUnsecuredLines']==0)].index, inplace=True)
median3 = train2.loc[train2['RevolvingUtilizationOfUnsecuredLines']<1, 'RevolvingUtilizationOfUnsecuredLines'].median()
train2.loc[train2['RevolvingUtilizationOfUnsecuredLines']> 1, 'RevolvingUtilizationOfUnsecuredLines'] = np.nan
train2['RevolvingUtilizationOfUnsecuredLines'].fillna(median,inplace=True)

#Feature Engineering
train2['PerPersonincome'] = train2['MonthlyIncome']/(train2['NumberOfDependents']+1)
test2['PerPersonincome'] = test2['MonthlyIncome']/(test2['NumberOfDependents']+1)

train2['Latefreq'] = train2['NumberOfTimes90DaysLate']+train2['NumberOfTime60-89DaysPastDueNotWorse'] +train2['NumberOfTime30-59DaysPastDueNotWorse']
test2['Latefreq'] = test2['NumberOfTimes90DaysLate']+test2['NumberOfTime60-89DaysPastDueNotWorse'] +test2['NumberOfTime30-59DaysPastDueNotWorse']

train2['MonthlyDebt'] = train2['DebtRatio']*train2['MonthlyIncome']
test2['MonthlyDebt'] = test2['DebtRatio']*test2['MonthlyIncome']

train2['NumOfOpenCreditLines'] = train2['NumberOfOpenCreditLinesAndLoans']-train2['NumberRealEstateLoansOrLines']
test2['NumOfOpenCreditLines'] = test2['NumberOfOpenCreditLinesAndLoans']-test2['NumberRealEstateLoansOrLines']

train2['MonthlyBal'] = train2['MonthlyIncome']-train2['MonthlyDebt']
test2['MonthlyBal'] = test2['MonthlyIncome']-test2['MonthlyDebt']

#Xgboost Model

def xgbCV(eta=[0.05],max_depth=[6],sub_sample=[0.9],colsample_bytree=[0.9]):
    train_y = train2['SeriousDlqin2yrs'] # label for training data
    train_X = train2.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False) # feature for training data
    test_X = test2.drop(['SeriousDlqin2yrs','ID'],axis=1,inplace=False) # feature for testing data
    skf = model_selection.StratifiedKFold(n_splits=5,random_state=100) # stratified sampling
    train_performance ={} 
    val_performance={}
    for each_param in itertools.product(eta,max_depth,sub_sample,colsample_bytree): # iterative over each combination in parameter space
        xgb_params = {
                    'eta':each_param[0],
                    'max_depth':each_param[1],
                    'sub_sample':each_param[2],
                    'colsample_bytree':each_param[3],
                    'objective':'binary:logistic',
                    'eval_metric':'auc',
                    'silent':0
                    }
        best_iteration =[]
        best_score=[]
        training_score=[]
        for train_ind,val_ind in skf.split(train_X,train_y): # five fold stratified cross validation
            X_train,X_val = train_X.iloc[train_ind,],train_X.iloc[val_ind,] # train X and train y
            y_train,y_val = train_y.iloc[train_ind],train_y.iloc[val_ind] # validation X and validation y
            dtrain = xgb.DMatrix(X_train,y_train,feature_names = X_train.columns) # convert into DMatrix (xgb library data structure)
            dval = xgb.DMatrix(X_val,y_val,feature_names = X_val.columns) # convert into DMatrix (xgb library data structure)
            model = xgb.train(xgb_params,dtrain,num_boost_round=1000, 
                              evals=[(dtrain,'train'),(dval,'val')],verbose_eval=False,early_stopping_rounds=30) # train the model
            best_iteration.append(model.attributes()['best_iteration']) # best iteration regarding AUC in valid set
            best_score.append(model.attributes()['best_score']) # best score regarding AUC in valid set
            training_score.append(model.attributes()['best_msg'].split()[1][10:]) # best score regarding AUC in training set
        valid_mean = (np.asarray(best_score).astype(np.float).mean()) # mean AUC in valid set
        train_mean = (np.asarray(training_score).astype(np.float).mean()) # mean AUC in training set
        val_performance[each_param] =  train_mean
        train_performance[each_param] =  valid_mean
        print ("Parameters are {}. Training performance is {:.4f}. Validation performance is {:.4f}".format(each_param,train_mean,valid_mean))
    return (train_performance,val_performance)
#xgbCV(eta=[0.01,0.02,0.03,0.04,0.05],max_depth=[4,6,8,10],colsample_bytree=[0.3,0.5,0.7,0.9]) 
xgbCV(eta=[0.04],max_depth=[4],colsample_bytree=[0.5])

print(train_X.columns)
any(train_X.columns == test_X.columns)

train_df = xgb.DMatrix(train_X,train_y,feature_names=train_X.columns)
test_df= xgb.DMatrix(test_X,feature_names=test_X.columns)
xgb_params = {
                    'eta':0.03,
                    'max_depth':4,
                    'sub_sample':0.9,
                    'colsample_bytree':0.5,
                    'objective':'binary:logistic',
                    'eval_metric':'auc',
                    'silent':0
                    }

final_model = xgb.train(xgb_params,train_df,num_boost_round=500)
ypredicted = final_model.predict(test_df)


xgb.plot_importance(final_model)
plt.show()

































