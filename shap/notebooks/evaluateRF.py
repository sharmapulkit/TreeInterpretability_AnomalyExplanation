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
from scipy.stats import spearmanr
import os

from dataLoader import *
from treeRegressor import *
import utils


epsilon = 0.00001

####### Import Output, Covariate and Treatment columns names for postgreSQL Datasetp
from postgresql_dataConfig import *
######################################################

def compute_ti_attribution(model, data):
	ti_start_time = time.time()
	ti_preds, ti_biases, ti_contribs = ti.predict(model, data)
	ti_end_time = time.time()
	print("Time for Tree Interpretation:", ti_end_time - ti_start_time)
	return ti_preds, ti_biases, ti_contribs, (ti_end_time - ti_start_time)

def compute_shap_attribution(model, data):
	shap_start_time = time.time()
	explainer = shap.TreeExplainer(model)
	shap_values = explainer.shap_values(data)
	shap_end_time = time.time()
	print("Time for SHAP explaination values:", shap_end_time - shap_start_time)
	return shap_values, (shap_end_time - shap_start_time)

def evaluateRF(modelPath, datasetPath, current_target, TrainValTest_split=(1.0,0.0,0.0), outdir=None):
	"""
	Train a Random Forest with given training dataset
	"""
	start_time = time.time()
	
	dataloader = DataLoader(datasetPath, covariate_columns, treatment_columns, target_columns)
	X, logY, _, _, _, _ = dataloader.preprocessData(train_frac=TrainValTest_split[0], val_frac=TrainValTest_split[1], test_frac=TrainValTest_split[2])
	data_preprocessed_time = time.time()
	print("Time to preprocess:", data_preprocessed_time - start_time)
	print("Test Data Size:", X.shape)
	treereg = TreeRegression(200, 20)

	print("Load Training model....")
	loaded_model = pk.load(open(modelPath, 'rb'))
	treereg.set_model(loaded_model)

	print("Test Evaluation....")
	testMSE, testR2 = treereg.evaluateRF(X, logY.loc[:, current_target])
	print("Test R2 Score:", testR2)

	if not outdir is None:
		with open(outdir, 'w') as f:
			f.write("MSE:"+ str(testMSE))
			f.write('\n')
			f.write("R2:"+ str(testR2))
			f.write('\n')


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Parser for training RF on dataset')

	parser.add_argument('--model_dir', help='Path to the trained model')
	parser.add_argument('--dataset_dir', help='Path to dataset csv file')
	parser.add_argument('--current_target', help='string label for current target label')
	parser.add_argument('--TrainValTest_split', help='Tuple with split ratios of dataset')
	parser.add_argument('--outdir', help='Dump the evaluation Metric to this file')

	args = vars(parser.parse_args())

	evaluateRF( args['model_dir'], args['dataset_dir'], args['current_target'],	make_tuple(args['TrainValTest_split']), outdir=args['outdir'])


