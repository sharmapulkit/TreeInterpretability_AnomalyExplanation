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
import argparse
from ast import literal_eval as make_tuple
from dataLoader import *


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
			rf.fit(X_train, Y_train_normalized.loc[:, current_target] )
			###### Get the accuracy of RandomForestRegression ########## 
			########### TRAINING SET ############
			Y_train_pred = rf.predict(X_train)
			print("Train Set MSE:", metrics.mean_absolute_error( \
			                  Y_train_normalized.loc[:, current_target] , Y_train_pred) )
			print("Train Set R2:", metrics.r2_score( (Y_train_pred), Y_train_normalized.loc[:, current_target] ))
			###### Get the accuracy of RandomForestRegression ##########
			########### TEST SET ############
			Y_test_pred = rf.predict(X_test)
			print("Test Set MSE:", metrics.mean_absolute_error( \
			                      Y_test_normalized.loc[:, current_target], Y_test_pred))
			r2_test = metrics.r2_score( Y_test_pred, Y_test_normalized.loc[:, current_target] )
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

