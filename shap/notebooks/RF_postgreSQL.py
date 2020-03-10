#!/usr/bin/env python
# coding: utf-8

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
import pickle as pk
import itertools
import time

# print the JS visualization code to the notebook

epsilon = 0.00001

class TreeRegression():
	def __init__(self, dataset_path):
		self.dataset_path = dataset_path
		

	####### Normalize columns of Y_train ########
	def normalize_columns(self, Y):
		x = data.values
		min_max_scaler = preprocessing.MinMaxScaler()
		Y_normalized = Y.copy()

		for target in target_columns:
		    col_min = Y_normalized.loc[:, target].min()
		    col_max = Y_normalized.loc[:, target].max()
		    Y_normalized[target] = (Y_normalized.loc[:, target] - col_min)/(col_max - col_min)
		
		return Y_normalized


	####### Log normalize targets ###########
	def log_normalize(self, data):
		logY_normalized = data.copy()
		logY_normalized = np.log(logY_normalized + epsilon)
		for target in list(data.columns):
		    col_min = logY_normalized.loc[:, target].min()
		    col_max = logY_normalized.loc[:, target].max()
		    logY_normalized[target]  = (logY_normalized.loc[:, target] - col_min)/(col_max - col_min)

		return logY_normalized


	######### Train Random Forest Regressor of the above data
	def trainRF(self, X_train, Y_train, current_target, rf_n_estimators=200, rf_max_depth=20):
		rf = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
		rf.fit(X_train, Y_train.loc[:, current_target])
		return rf


	###### Get the accuracy of RandomForestRegression ########## 
	########### TRAINING SET ############
	def evaluateRF(self, X, Y_gt, model, current_target):
		current_target_id = 5
		Y_pred = model.predict(X)
		print(Y_pred.shape)
		print(Y_gt.shape)
		print("Set MSE:", metrics.mean_absolute_error( Y_gt, Y_pred ) )
		print("Set R2:", metrics.r2_score( Y_gt, Y_pred ))


	def grid_search_rf_parameters(self, X_train, Y_train, X_test, Y_test, current_target):
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
			print("Train Set MSE:", metrics.mean_absolute_error( \
			                  np.log(Y_train_normalized.loc[:, current_target] + epsilon), Y_train_pred) )
			print("Train Set R2:", metrics.r2_score( (Y_train_pred),                     np.log(Y_train_normalized.loc[:, current_target] + epsilon) ))
			###### Get the accuracy of RandomForestRegression ##########
			########### TEST SET ############
			Y_test_pred = rf.predict(X_test)
			print("Test Set MSE:", metrics.mean_absolute_error( \
			                      np.log(Y_test_normalized.loc[:, current_target] + epsilon), Y_test_pred))
			r2_test = metrics.r2_score((Y_test_pred),                         np.log(Y_test_normalized.loc[:, current_target] + epsilon))
			print("Test Set R2:",  r2_test)
			if (best_combo_r2_test < r2_test):
				best_combo_r2_test = r2_test
				best_combo = (num_ests, m_depth_it)


		print("Best found R2:", best_combo_r2_test)
		print("Best found (N_estimators, max_depth):", best_combo)
		return(best_combo, best_combo_r2_test)


	########## Train XGBoost ##########
	def trainXGB(self, X_train, Y_train, X_test, Y_test):
		########### Prepare data for XGBoost ############
		Y_range = np.max(logY_normalized.loc[:, current_target]) - np.min(logY_normalized.loc[:, current_target])
		d_train = xgboost.DMatrix(X_train, label=logY_train_normalized.loc[:, current_target] )
		d_test  = xgboost.DMatrix(X_test , label=logY_test_normalized.loc[:, current_target] )
		base_score = np.mean(Y_train.extend(Y_test))
		
		######### Train XGBoost on dataset ###########
		params = {
		    "eta": 0.01,
		    "objective": "reg:squarederror",
		    "subsample": 0.5,
		    "base_score": np.mean(logY_train_normalized.loc[:, current_target]),
		    "eval_metric": "logloss"
		}
		model = xgboost.train(params, d_train, 5000, evals = [(d_test, "test")], verbose_eval=100, early_stopping_rounds=50) 

	def inferXGB(self, d_test):
		###### Inference on XGBoost trained models #########
		logY_pred_xgb_test = model.predict(d_test)
		r2_test_xgb = metrics.r2_score(logY_pred_xgb_test, logY_test_normalized.loc[:, current_target] )
		print("XGB R2 Test :", r2_test_xgb)


	def normalize_columns(self, Y_train):
		## Normalize columns of Y_train
		x = data.values
		min_max_scaler = preprocessing.MinMaxScaler()
		Y_normalized = data.copy()

		for target in list(Y_train.columns):
		    col_min = Y_normalized.loc[:, target].min()
		    col_max = Y_normalized.loc[:, target].max()
		    Y_normalized[target] = (Y_normalized.loc[:, target] - col_min)/(col_max - col_min)

		Y_train_normalized = Y_normalized[:train_size]
		Y_test_normalized = Y_normalized[train_size:]
		return (Y_train_normalized, Y_test_normalized)


from scipy.stats import spearmanr
def print_spearmanr(ranking1, ranking2):
	coeffs, ps = [], []
	for r1, r2 in zip(ranking1, ranking2):
		coef, p = spearmanr(r1, r2)
		coeffs.append(coef)
		ps.append(p)
	print(pd.Series(coeffs).describe())


if __name__=="__main__":
	#### Dataset path ####
	PATH='~/cs696ds/TreeInterpretability_AnomalyExplanation/'
	print("Loading dataset...")
	data = pd.read_csv(PATH+"datasets/postgres-results.csv")
	print(data.shape)

	all_feats = list(data)
	
	####### Define Output, Covariate and Treatment columns names
	target_columns = ['local_written_blocks', 'temp_written_blocks', 'shared_hit_blocks', 'temp_read_blocks', 'local_read_blocks', 'runtime', 'shared_read_blocks']
	treatment_columns = ['index_level', 'page_cost', 'memory_level']
	covariate_columns = ['rows', 'creation_year', 'num_ref_tables', 'num_joins', 'num_group_by', 'queries_by_user', 'length_chars', 'total_ref_rows', 'local_hit_blocks', 'favorite_count']
	feature_columns = covariate_columns
	feature_columns.extend(treatment_columns)

	train_size = int(data.shape[0]*0.7)
	X_train = data[feature_columns][:train_size]
	Y_train = data[target_columns][:train_size]

	X_test = data[feature_columns][train_size:]
	Y_test = data[target_columns][train_size:]

	treereg = TreeRegression(PATH)
	
	logY_train_normalized = treereg.log_normalize(Y_train)
	logY_test_normalized  = treereg.log_normalize(Y_test)

	current_target = 'runtime'
	rf_n_estimators = 200
	rf_max_depth = 20

	SAVE_RF = False
	model_filename = "RF_postgres_Nest{}_maxD{}_{}".format(rf_n_estimators, rf_max_depth, 'runtime')
	data_preprocessed_time = time.time()
	if (SAVE_RF):
		print("Training Random Forest...")
		rf = treereg.trainRF(X_train, logY_train_normalized, current_target, rf_n_estimators, rf_max_depth)
		rf_trained_time = time.time()
		pk.dump(rf, open(model_filename, 'wb'))
		model_file_dumped = time.time()
		print("Time to Train RF:", rf_trained_time - data_preprocessed_time)
	else:
		rf = pk.load(open(model_filename, 'rb'))	
		loaded_model = time.time()
		print("Time to load model:", loaded_model - data_preprocessed_time)

	#print("Train Evaluation....")
	#treereg.evaluateRF(X_train, logY_train_normalized, rf, current_target)
	#print(Y_test.isna().sum())
	#print(logY_test_normalized.isna().sum())
	print("Test Evaluation....")
	#treereg.evaluateRF(X_test, logY_test_normalized, rf, current_target)
	#print(X_test.shape)
	#print(logY_test_normalized.loc[:, current_target].shape)
	treereg.evaluateRF(X_test, Y_test.loc[:, current_target], rf, current_target)
	treereg.evaluateRF(X_test, logY_test_normalized.loc[:, current_target], rf, current_target)
	print("Evaluation time:", time.time() - loaded_model)
	
	evaluated_time = time.time()
	ti_preds, ti_biases, ti_contribs = ti.predict(rf, X_test[:500])
	ti_values_time = time.time()
	print("Time for Tree Interpretation:", ti_values_time - evaluated_time)
	
	explainer = shap.TreeExplainer(rf)
	shap_values = explainer.shap_values(X_test[:500])
	shap_values_time = time.time()
	print("Time for SHAP explaination values:", shap_values_time - ti_values_time)
	
	print(ti_contribs.shape)
	print(shap_values.shape)
	print_spearmanr(ti_contribs, shap_values)
	
