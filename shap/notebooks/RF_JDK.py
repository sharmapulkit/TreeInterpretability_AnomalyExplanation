#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import xgboost
import shap
import pandas as pd
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from treeinterpreter import treeinterpreter as ti
from sklearn import metrics

# print the JS visualization code to the notebook
shap.initjs()


PATH='~/Documents/PDF/umass/sem2/cs696ds/code/'
data = pd.read_csv(PATH+"datasets/jdk-results.csv")


# # Data Preprocessing

all_feats = list(data)
epsilon = 0.00001


target_columns = ['num_bytecode_ops', 'total_unit_test_time', 'allocated_bytes', 'jar_file_size_bytes', 'compile_time_ms' ]
treatment_columns = ['debug', 'obfuscate', 'parallelgc']
covariate_columns = ['source_ncss', 'test_classes', 'test_functions', 'test_ncss', 'test_javadocs']
feature_columns = covariate_columns
feature_columns.extend(treatment_columns)


train_size = int(data.shape[0]*0.7)
X_train = data[feature_columns][:train_size]
Y_train = data[target_columns][:train_size]

X_test = data[feature_columns][train_size:]
Y_test = data[target_columns][train_size:]



## Normalize columns of Y_train
x = data.values
min_max_scaler = preprocessing.MinMaxScaler()
Y_normalized = data[target_columns].copy()

for target in target_columns:
    col_min = Y_normalized.loc[:, target].min()
    col_max = Y_normalized.loc[:, target].max()
    Y_normalized[target] = (Y_normalized.loc[:, target] - col_min)/(col_max - col_min)

# print(Y_normalized.loc[:, 'html_attrs'])
# print(np.max(Y_train.loc[:, target]) - np.min(Y_train[:, target]))
Y_train_normalized = Y_normalized[:train_size]
Y_test_normalized = Y_normalized[train_size:]


# In[154]:


####### Log normalize targets ###########
logY_normalized = data[target_columns].copy()
logY_normalized = np.log(logY_normalized + epsilon)
for target in target_columns:
    col_min = logY_normalized.loc[:, target].min()
    col_max = logY_normalized.loc[:, target].max()
    logY_normalized[target]  = (logY_normalized.loc[:, target] - col_min)/(col_max - col_min)

logY_train_normalized = logY_normalized[:train_size]
logY_test_normalized = logY_normalized[train_size:]


# In[27]:


current_target = 'compile_time_ms'


# # Train Random Forest Regressor of the above data

# In[155]:


rf = RandomForestRegressor(n_estimators=1600, max_depth=160)
rf.fit(X_train, logY_train_normalized.loc[:, current_target] )


# In[156]:


###### Get the accuracy of RandomForestRegression ########## 
########### TRAINING SET ############
Y_train_pred = rf.predict(X_train)
print("Train Set MSE:", metrics.mean_absolute_error(                 logY_train_normalized.loc[:, current_target], Y_train_pred) )
print("Train Set R2:", metrics.r2_score( (Y_train_pred),                 logY_train_normalized.loc[:, current_target] ))


# In[158]:


###### Get the accuracy of RandomForestRegression ##########
########### TEST SET ############
Y_test_pred = rf.predict(X_test)
print("Test Set MSE:", metrics.mean_absolute_error(                     logY_test_normalized.loc[:, current_target], Y_test_pred))
print("Test Set R2:", metrics.r2_score((Y_test_pred),                     logY_test_normalized.loc[:, current_target]))


# In[113]:


import itertools
# for x in list(itertools.product(range(1200, 2201, 100), range(0, 100, 10))):
#     print(x)


# In[116]:


########## Grid Search for RF Training parameters
best_combo = (1200, 100)
best_combo_r2_test = -90000
for (num_ests, m_depth_it) in itertools.product(range(1200, 2201, 100), range(100, 200, 10)):
    print("Number of Estimators:", num_ests, " - Max Depth:",m_depth_it)
    rf = RandomForestRegressor(n_estimators=num_ests, max_depth=m_depth_it)
    rf.fit(X_train, np.log(Y_train_normalized.loc[:, current_target] + epsilon))
    ###### Get the accuracy of RandomForestRegression ########## 
    ########### TRAINING SET ############
    Y_train_pred = rf.predict(X_train)
#     print("Train Set MSE:", metrics.mean_absolute_error( \
#                     np.log(Y_train_normalized.loc[:, current_target] + epsilon), Y_train_pred) )
    print("Train Set R2:", metrics.r2_score( (Y_train_pred),                     np.log(Y_train_normalized.loc[:, current_target] + epsilon) ))
    ###### Get the accuracy of RandomForestRegression ##########
    ########### TEST SET ############
    Y_test_pred = rf.predict(X_test)
#     print("Test Set MSE:", metrics.mean_absolute_error( \
#                         np.log(Y_test_normalized.loc[:, current_target] + epsilon), Y_test_pred))
    r2_test = metrics.r2_score((Y_test_pred),                         np.log(Y_test_normalized.loc[:, current_target] + epsilon))
    print("Test Set R2:",  r2_test)
    if (best_combo_r2_test < r2_test):
        best_combo_r2_test = r2_test
        best_combo = (num_ests, m_depth_it)


# In[118]:


print(best_combo_r2_test)
print(best_combo)


# ### Train XGBoost

# In[164]:


########### Prepare data for XGBoost ############
Y_range = np.max(logY_normalized.loc[:, current_target])             - np.min(logY_normalized.loc[:, current_target])
d_train = xgboost.DMatrix(X_train, label=logY_train_normalized.loc[:, current_target] )
d_test  = xgboost.DMatrix(X_test , label=logY_test_normalized.loc[:, current_target] )


# In[166]:


######### Train XGBoost on dataset ###########
params = {
    "eta": 0.01,
    "objective": "reg:squarederror",
    "subsample": 0.5,
    "base_score": np.mean(logY_train_normalized.loc[:, current_target]),
    "eval_metric": "logloss"
}
model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")],                       verbose_eval=100, early_stopping_rounds=50) 


# In[163]:


###### Inference on XGBoost trained models #########
logY_pred_xgb_train = model.predict(d_train)
r2_train_xgb = metrics.r2_score(logY_pred_xgb_train,                         logY_train_normalized.loc[:, current_target] )
logY_pred_xgb_test = model.predict(d_test)
r2_test_xgb = metrics.r2_score(logY_pred_xgb_test,                         logY_test_normalized.loc[:, current_target] )
print("XGB R2 Train:", r2_train_xgb)
print("XGB R2 Test :", r2_test_xgb)


# In[ ]:


current_target = 'html_attrs'
##### Random Forest for html_attrs ####
rf_htmlattrs = RandomForestRegressor(n_estimators=100, max_depth=20)
# print(Y_train_normalized.loc[:, 'html_attrs'])
rf_htmlattrs.fit(X_train, Y_train_normalized.loc[:, current_target])
###### Get the accuracy of RandomForestRegression
Y_test_pred = rf_htmlattrs.predict(X_test)
print(metrics.mean_absolute_error(Y_test_normalized.loc[:, current_target], Y_test_pred))
print(metrics.r2_score(Y_test_normalized.loc[:, current_target], Y_test_pred))


# In[7]:


current_target = 'html_tags'
### Random Forest for html_tags
rf_htmltags = RandomForestRegressor(n_estimators=100, max_depth=20)
# print(Y_train_normalized.loc[:, 'html_attrs'])
rf_htmltags.fit(X_train, Y_train_normalized.loc[:, current_target])
###### Get the accuracy of RandomForestRegression
Y_test_pred = rf_htmltags.predict(X_test)
print(metrics.mean_absolute_error(Y_test_normalized.loc[:, current_target], Y_test_pred))
print(metrics.r2_score(Y_test_normalized.loc[:, current_target], Y_test_pred))


# In[ ]:


current_target = 'elapsed'
##### Random Forest for elapsed ####
rf_elapsed = RandomForestRegressor(n_estimators=100, max_depth=20)
# print(Y_train_normalized.loc[:, 'html_attrs'])
rf_elapsed.fit(X_train, Y_train_normalized.loc[:, current_target])
###### Get the accuracy of RandomForestRegression
Y_test_pred = rf_elapsed.predict(X_test)
print(metrics.mean_absolute_error(Y_test_normalized.loc[:, current_target], Y_test_pred))
print(metrics.r2_score(Y_test_normalized.loc[:, current_target], Y_test_pred))


# # Train using output variables

# In[25]:


target_columns = ['elapsed', '']
feature_columns = ['mobile_user_agent', 'proxy', 'compression',                    'trial', 'ping_time', 'html_tags', 'html_attrs', 'decompressed_content_length',                   'raw_content_length']
covariate_columns = ['record.count']
treatment_columns = ['mobile_user_agent', 'proxy', 'compression']


# In[26]:


train_size = int(data.shape[0]*0.7)
X_train = data[feature_columns][:train_size]
Y_train = data[target_columns][:train_size]

X_test = data[feature_columns][train_size:]
Y_test = data[target_columns][train_size:]


# In[ ]:


## Normalize columns of Y_train
x = data.values
min_max_scaler = preprocessing.MinMaxScaler()
Y_normalized = data[target_columns].copy()

for target in target_columns:
    col_min = Y_normalized.loc[:, target].min()
    col_max = Y_normalized.loc[:, target].max()
    Y_normalized[target] = (Y_normalized.loc[:, target] - col_min)/(col_max - col_min)

# print(Y_normalized.loc[:, 'html_attrs'])
# print(np.max(Y_train.loc[:, target]) - np.min(Y_train[:, target]))
Y_train_normalized = Y_normalized[:train_size]
Y_test_normalized = Y_normalized[train_size:]


# ### Train a model for Elapsed using output variables

# In[28]:


current_target = 'elapsed'
##### Random Forest for Elapsed ####
rf_elapsed = RandomForestRegressor(n_estimators=100, max_depth=20)
# print(Y_train_normalized.loc[:, 'html_attrs'])
rf_elapsed.fit(X_train, Y_train_normalized.loc[:, current_target])


# In[29]:


###### Get the accuracy of RandomForestRegression
Y_test_pred = rf_elapsed.predict(X_test)
print("MSE:",metrics.mean_absolute_error(Y_test_normalized.loc[:, current_target], Y_test_pred))
print("R2:",metrics.r2_score(Y_test_normalized.loc[:, current_target], Y_test_pred))


# ### Train a model for html_attrs using output variables

# In[30]:


current_target = 'html_attrs'
##### Random Forest for html_attrs ####
rf_htmlattrs = RandomForestRegressor(n_estimators=100, max_depth=20)
rf_htmlattrs.fit(X_train, Y_train_normalized.loc[:, current_target])


# In[31]:


###### Get the accuracy of RandomForestRegression
Y_test_pred = rf_htmlattrs.predict(X_test)
print("MSE:",metrics.mean_absolute_error(Y_test_normalized.loc[:, current_target], Y_test_pred))
print("R2:",metrics.r2_score(Y_test_normalized.loc[:, current_target], Y_test_pred))


# # Analyse Feature Importance

# In[18]:


rf = rf_htmlattrs


# In[19]:


### Global feature Importances
print("Feature Importances:", rf.feature_importances_)

### Local feature Importances
instance = X_train.loc[0:10, :]
prediction, bias, contributions = ti.predict(rf, instance)
# print(prediction, bias)
print("Contributions :", contributions)


# In[20]:


explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train.loc[:10, :])
print(shap_values)


# In[23]:


shap.force_plot(explainer.expected_value, shap_values[:50, :], X_train.iloc[:50, :])


# In[19]:


from scipy.stats import spearmanr
coef, p = spearmanr(shap_values[0, :], contributions[0, :])
print(coef, p)


# In[25]:


a = [1, 4, 3, 2, 1]
b = [1, 4, 2, 3, 8]
c, pp = spearmanr(a, b)
print(c, pp)


# In[38]:


import lime
import lime.lime_tabular
lime.lime_tabular.LimeTabularExplainer(X_train, Y_train_normalized)

