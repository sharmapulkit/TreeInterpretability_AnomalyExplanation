#!/usr/bin/env python
# coding: utf-8

import numpy as np
import csv
import pandas as pd
from sklearn import metrics
import pickle as pk
import itertools
import time
import argparse
from ast import literal_eval as make_tuple


epsilon = 0.00001

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
		_data_copy = self._data[(self._data.loc[:, self._treatment_columns[0]] == t_comb[0]) & \
					  (self._data.loc[:, self._treatment_columns[1]] == t_comb[1]) & \
					  (self._data.loc[:, self._treatment_columns[2]] == t_comb[2])]
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
			data_tr = data[(data.loc[:, self._treatment_columns[0]] == t_comb[0]) & \
					  (data.loc[:, self._treatment_columns[1]] == t_comb[1]) & \
					  (data.loc[:, self._treatment_columns[2]] == t_comb[2])]
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

