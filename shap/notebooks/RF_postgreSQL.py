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

############## DEFINE MACRO VARIABLES #############
#### Dataset path ####
PATH='/home/s20psharma/cs696ds/TreeInterpretability_AnomalyExplanation/'
print("Loading dataset...")
data = pd.read_csv(PATH+"datasets/postgres-results.csv")
print(data.shape)

all_feats = list(data)

####### Define Output, Covariate and Treatment columns names
target_columns 	  = ['local_written_blocks', 'temp_written_blocks', 'shared_hit_blocks', 'temp_read_blocks', 'local_read_blocks', 'runtime', 'shared_read_blocks']
treatment_columns = ['index_level', 'page_cost', 'memory_level']
covariate_columns = ['rows', 'creation_year', 'num_ref_tables', 'num_joins', 'num_group_by', 'queries_by_user', 'length_chars', 'total_ref_rows', 'local_hit_blocks', 'favorite_count']
feature_columns   = covariate_columns.copy()
feature_columns.extend(treatment_columns)
######################################################


class DataLoader():
	def __init__(self, dataset_path, covariate_columns = [], treatment_columns = [], target_columns = []):
		self._dataset_path = dataset_path
		self._covariate_columns = covariate_columns
		self._treatment_columns = treatment_columns
		self._target_columns = target_columns
		self._feature_columns = self._covariate_columns.copy()
		self._feature_columns.extend(treatment_columns)
		self._data = pd.read_csv(self._dataset_path)

	####### Normalize columns ########
	def normalize_columns(self, columns):
		for c in columns:
		    col_min = self._data.loc[:, target].min()
		    col_max = self._data.loc[:, target].max()
		    Y_normalized[c] = (Y_normalized.loc[:, c] - col_min)/(col_max - col_min)
		return Y_normalized

	####### Log normalize all columns ###########
	def logNormalizeTargets(self):
		# self._data = np.log(self._data + epsilon)
		for target in list(self._target_columns):
			self._data.loc[:, target] = np.log(self._data.loc[:, target] + epsilon)
			# print(self._data.shape)
			col_min = self._data.loc[:, target].min()
			col_max = self._data.loc[:, target].max()
			self._data[target]  = (self._data.loc[:, target] - col_min) / (col_max - col_min)
		return

	def preprocessData(self, train_frac=0.7, val_frac=0.0, test_frac=0.3):
		train_size = int(data.shape[0]*train_frac)
		val_size = int(data.shape[0]*val_frac)
		test_size = int(data.shape[0]*test_frac)
		self.logNormalizeTargets()
		
		X_train = self._data[self._feature_columns][:train_size]
		logYtrain_normalized = self._data[self._target_columns][:train_size]

		X_val = self._data[self._feature_columns][train_size:train_size + val_size]
		logYval_normalized = self._data[self._target_columns][train_size:train_size + val_size]

		X_test = self._data[self._feature_columns][train_size + val_size:]
		logYtest_normalized = self._data[self._target_columns][train_size + val_size:]

		return X_train, logYtrain_normalized, X_val, logYval_normalized, X_test, logYtest_normalized

	def preprocessDataTreatmentCombo(self, treatmentCombo=(0,0,0), train_frac=0.7, val_frac=0.0, test_frac=0.3):
		self.logNormalizeTargets()
		t_comb = treatmentCombo
		_data_copy = self._data[(self._data.loc[:, treatment_columns[0]] == t_comb[0]) & \
					  (self._data.loc[:, treatment_columns[1]] == t_comb[1]) & \
					  (self._data.loc[:, treatment_columns[2]] == t_comb[2])]
		train_size = int(_data_copy.shape[0]*train_frac)
		val_size = int(_data_copy.shape[0]*val_frac)
		test_size = int(_data_copy.shape[0]*test_frac)

		treatX_train = _data_copy[self._feature_columns][:train_size]
		treatY_train = _data_copy[self._target_columns][:train_size]

		treatX_val = _data_copy[self._feature_columns][train_size:train_size + val_size]
		treatY_val = _data_copy[self._target_columns][train_size:train_size + val_size]

		treatX_test = _data_copy[self._feature_columns][train_size+val_size:]
		treatY_test = _data_copy[self._target_columns][train_size+val_size:]

		return treatX_train, treatY_train, treatX_val, treatY_val, treatX_test, treatY_test



	def preprocessDataTreatment(self, train_frac=0.7, val_frac=0.0, test_frac=0.3):
		train_size = int(self._data.shape[0]*train_frac)
		val_size = int(self._data.shape[0]*val_frac)
		test_size = int(self._data.shape[0]*test_frac)

		l = [0, 1, 2]
		treatment_combinations = list(itertools.product(l, repeat=3))
		self.logNormalizeTargets()
		X_trains, Y_trains, X_vals, Y_vals, X_tests, Y_tests = [], [], [], [], [], []
		for t_comb in treatment_combinations:
			data_tr = data[(data.loc[:, treatment_columns[0]] == t_comb[0]) & \
					  (data.loc[:, treatment_columns[1]] == t_comb[1]) & \
					  (data.loc[:, treatment_columns[2]] == t_comb[2])]
			treatX_train = self._data[self._feature_columns][:train_size]
			treatY_train = self._data[self._target_columns][:train_size]

			treatX_val = self._data[self._feature_columns][train_size:train_size + val_size]
			treatY_val = self._data[self._target_columns][train_size:train_size + val_size]

			treatX_test = self._data[self._feature_columns][train_size+val_size:]
			treatY_test = self._data[self._target_columns][train_size+val_size:]

			X_trains.append(treatX_train)
			Y_trains.append(treatY_train)
			X_vals.append(treatX_val)
			Y_vals.append(treatY_val)
			X_tests.append(treatX_test)
			Y_tests.append(treatY_test)

		return X_trains, Y_trains, X_vals, Y_vals, X_tests, Y_tests, treatment_combinations

class TreeRegression():
	def __init__(self, n_estimators=10, max_depth=10):
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.model = RandomForestRegressor(n_estimators, max_depth)
		
	######### Getter ##########
	def get_model(self):
		return self.model

	######### Setter ##########
	def set_nEst(self, n_est):
		self.n_estimators = n_est
	def set_maxDepth(self, max_D):
		self.max_depth = max_D

	######### Train Random Forest Regressor
	def trainRF(self, X_train, Y_train, current_target, rf_n_estimators=200, rf_max_depth=20):
		self.model = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth)
		self.model.fit(X_train, Y_train.loc[:, current_target])
		return

	def inferRF(self, X_test):
		Y_preds = self.model.predict(X_test)
		return Y_preds

	###### Get the accuracy of RandomForestRegression ########## 
	########### TRAINING SET ############
	def evaluateRF(self, Xtest, Ytest):
		Ypred = self.model.predict(Xtest)
		mse = metrics.mean_absolute_error(Ytest, Ypred)
		r2 = metrics.r2_score(Ytest, Ypred)
		return (mse, r2)

	def grid_search_rf_depth(self, X_train, Y_train, X_val, Y_val, current_target, num_est=200):
		m_depth_values = [10, 20, 40, 80, 160]
		best_r2_val_value = -100000
		best_maxDepth = 10
		for m_depth in m_depth_values:
			print("Max Depth:", m_depth)
			self.trainRF(X_train, Y_train, current_target, num_est, m_depth)
			###### Get the accuracy of RandomForestRegression ########## 
			########### TRAINING SET ############
			Y_train_pred = self.inferRF(X_train)
			print("Train Set MSE:", metrics.mean_absolute_error( \
			                  Y_train.loc[:, current_target], Y_train_pred) )
			print("Train Set R2:", metrics.r2_score( Y_train_pred, Y_train.loc[:, current_target]) )
			###### Get the accuracy of RandomForestRegression ##########
			########### TEST SET ############
			Y_val_pred = self.inferRF(X_val)
			print("Test Set MSE:", metrics.mean_absolute_error( \
			                      Y_val.loc[:, current_target], Y_val_pred))
			r2_test = metrics.r2_score( Y_val_pred, Y_val.loc[:, current_target] )
			print("Test Set R2:",  r2_test)
			if (best_r2_val_value < r2_test):
				best_r2_val_value = r2_test
				best_maxDepth = m_depth

		print("Best found R2:", best_r2_val_value)
		print("Best found max_depth:", best_maxDepth)
		return (best_maxDepth, best_r2_val_value)
			

	def grid_search_rf_parameters(self, X_train, Y_train, X_val, Y_val, current_target):
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
			print("Train Set R2:", metrics.r2_score( (Y_train_pred), np.log(Y_train_normalized.loc[:, current_target] + epsilon) ))
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
		return (best_combo, best_combo_r2_test)


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



from scipy.stats import spearmanr
def print_spearmanr(ranking1, ranking2):
	coeffs, ps = [], []
	for r1, r2 in zip(ranking1, ranking2):
		coef, p = spearmanr(r1, r2)
		coeffs.append(coef)
		ps.append(p)
	print(pd.Series(coeffs).describe())

####### Log normalize targets ###########
def log_normalize(self, data):
	logData_normalized = data.copy()
	logData_normalized = np.log(logData_normalized + epsilon)
	for col in list(data.columns):
	    col_min = logData_normalized.loc[:, col].min()
	    col_max = logData_normalized.loc[:, col].max()
	    logData_normalized[col]  = (logData_normalized.loc[:, col] - col_min)/(col_max - col_min)
	return logData_normalized


def mainTreatments():
	### Training parameters
	SAVE_RF = True
	outdir = PATH + "shap/notebooks/trainedNetworks/"
	current_target = 'runtime'
	rf_n_estimators = 200
	rf_max_depth = 20

	treereg = TreeRegression(PATH)		
	start_time = time.time()
	dataloader = DataLoader(PATH + "datasets/postgres-results.csv", covariate_columns, treatment_columns, target_columns)
 
	# X_trains, logY_train_normalizeds, X_vals, logY_vals, X_tests, logY_test_normalizeds, tr_combos = dataloader.preprocessDataTreatmentCombo((0,0,0), train_frac=0.1, val_frac=0.0, test_frac=0.1)
	l = [0, 1, 2]
	for tr_comb in list(itertools.product(l, repeat=3)):
	#tr_comb = (0, 0, 0)
		print("Treatment Combination:", tr_comb)
		Xtr, logYtr, Xval, logYval, Xte, logYte = dataloader.preprocessDataTreatmentCombo(tr_comb, train_frac=0.6, val_frac=0.2, test_frac=0.2)
		print("Train Size:", Xtr.shape)
		print("Val Size:", Xval.shape)
		print("Test Size:", Xte.shape)

		data_preprocessed_time = time.time()
		print("Time to preprocess:", data_preprocessed_time - start_time)
		#treereg.grid_search_rf_depth(Xtr, logYtr, Xval, logYval, current_target)

		# for (Xtr, logYtr, Xval, logYval, Xte, logYte, tr_comb) in zip(X_trains, logY_train_normalizeds, X_vals, logY_vals, X_tests, logY_test_normalizeds, tr_combos):
		# Xtr, logYtr, Xval, logYval, Xte, logYte, tr_comb = X_trains[0], logY_train_normalizeds[0], X_vals[0], logY_vals[0], X_tests[0], logY_test_normalizeds[0], tr_combos[0]
		model_filename = outdir + "RF_postgres_Nest{}_maxD{}_{}_tr{}{}{}".format(rf_n_estimators, rf_max_depth, current_target, tr_comb[0], tr_comb[1], tr_comb[2])
		if (SAVE_RF):
			print("Training Random Forest...")
			treereg.trainRF(Xtr, logYtr, current_target, rf_n_estimators, rf_max_depth)
			rf = treereg.get_model()
			rf_trained_time = time.time()
			print("Time to Train RF:", rf_trained_time - data_preprocessed_time)
			pk.dump(rf, open(model_filename, 'wb'))
			loaded_model_timestamp = time.time()
		else:
			rf = pk.load(open(model_filename, 'rb'))	
			loaded_model_timestamp = time.time()
			print("Time to load model:", loaded_model_timestamp - data_preprocessed_time)

		print("Train Evaluation....")
		trainMSE, trainR2 = treereg.evaluateRF(Xtr, logYtr.loc[:, current_target])
		print("Train R2 Score:", trainR2)
		print("Test Evaluation....")
		testMSE, testR2 = treereg.evaluateRF(Xte, logYte.loc[:, current_target])
		print("Test R2 Score:", testR2)
		print("Evaluation time:", time.time() - loaded_model_timestamp)
		
		evaluated_time = time.time()
		ti_preds, ti_biases, ti_contribs = ti.predict(rf, Xte[:500])
		ti_values_time = time.time()
		print("Time for Tree Interpretation:", ti_values_time - evaluated_time)
		
		explainer = shap.TreeExplainer(rf)
		shap_values = explainer.shap_values(Xte[:500])
		shap_values_time = time.time()
		print("Time for SHAP explaination values:", shap_values_time - ti_values_time)
		
		print(ti_contribs.shape)
		print(shap_values.shape)
		print_spearmanr(ti_contribs, shap_values)

def main():
	treereg = TreeRegression(PATH)		
	start_time = time.time()
	X_trains, logY_train_normalizeds, X_tests, logY_test_normalizeds = preprocessDataTreatment(data, feature_columns, target_columns, treatment_columns)
	data_preprocessed_time = time.time()
	print("Time to preprocess:", data_preprocessed_time - start_time)

	current_target = 'runtime'
	rf_n_estimators = 200
	rf_max_depth = 10

	SAVE_RF = True
	model_filename = "RF_postgres_Nest{}_maxD{}_{}".format(rf_n_estimators, rf_max_depth, 'runtime')
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

if __name__=="__main__":
	mainTreatments()
