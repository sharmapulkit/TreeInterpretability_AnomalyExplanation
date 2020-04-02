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

def trainRF(SAVE_RF, outdir, datasetPath, current_target, rf_n_estimators, rf_max_depth, TrainValTest_split=(0.6,0.2,0.2)):
	"""
	Train a Random Forest with given training dataset
	"""
	start_time = time.time()
	timing_info = []
	if os.path.exists(outdir):
		print("File Exists at given outdir. Rename outdir.")
		return
	
	dataloader = DataLoader(datasetPath, covariate_columns, treatment_columns, target_columns)
	Xtr, logYtr, _, _, _, _ = dataloader.preprocessData(train_frac=TrainValTest_split[0], val_frac=TrainValTest_split[1], test_frac=TrainValTest_split[2])
	data_preprocessed_time = time.time()
	print("Time to preprocess:", data_preprocessed_time - start_time)
	print("Train Size:", Xtr.shape)
	treereg = TreeRegression(rf_n_estimators, rf_max_depth)

	if (SAVE_RF):
		print("Training Random Forest...")
		treereg.trainRF(Xtr, logYtr, current_target, rf_n_estimators, rf_max_depth)
		rf = treereg.get_model()
		rf_trained_time = time.time()
		print("Time to Train RF:", rf_trained_time - data_preprocessed_time)

		pk.dump(rf, open(outdir, 'wb'))
		loaded_model_timestamp = time.time()


def mainTreatments(SAVE_RF, outdir, datasetPath, current_target, rf_n_estimators, rf_max_depth, timing_info_outfile, tr_comb=None, TrainValTest_split=(0.6,0.2,0.2)):
	start_time = time.time()
	timing_info = []
	
	dataloader = DataLoader(datasetPath, covariate_columns, treatment_columns, target_columns)
	if (tr_comb is None):
		Xtr, logYtr, Xval, logYval, Xte, logYte = dataloader.preprocessData(train_frac=TrainValTest_split[0], val_frac=TrainValTest_split[1], test_frac=TrainValTest_split[2])
	else:
		Xtr, logYtr, Xval, logYval, Xte, logYte = dataloader.preprocessDataTreatmentCombo(tr_comb, train_frac=TrainValTest_split[0], val_frac=TrainValTest_split[1], test_frac=TrainValTest_split[2])
	data_preprocessed_time = time.time()
	print("Time to preprocess:", data_preprocessed_time - start_time)

	print("Treatment Combination:", tr_comb)
	print("Train Size:", Xtr.shape)
	print("Val Size:",	 Xval.shape)
	print("Test Size:",  Xte.shape)
	treereg = TreeRegression(rf_n_estimators, rf_max_depth)

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
	evaluationTimestamp = time.time()
	print("Evaluation time:", evaluationTimestamp - loaded_model_timestamp)
	
	ti_preds, ti_biases, ti_contribs, ti_runtime = compute_ti_attribution(rf, Xte[:500])
	shap_values, shap_runtime = compute_shap_attribution(rf, Xte[:500])
	print(ti_contribs.shape)
	print(shap_values.shape)
	utils.print_spearmanr(ti_contribs, shap_values)
	
	##### Write the comparison data to a file
	timing_info.append(data_preprocessed_time - start_time)
	timing_info.append(rf_trained_time - data_preprocessed_time)
	timing_info.append(loaded_model_timestamp - data_preprocessed_time)
	timing_info.append(evaluationTimestamp - loaded_model_timestamp)
	timing_info.append(ti_values_time - evaluated_time)
	timing_info.append(shap_values_time - ti_values_time)
	utils.write_timing_info_file(timing_info_outfile, timing_info)


def main( SAVE_RF, outdir, datasetPath, current_target, rf_n_estimators, rf_max_depth, timing_info_outfile, TrainValTest_split=(0.6,0.2,0.2) ):
	timing_info = []
	start_time = time.time()
	
	dataloader = DataLoader(datasetPath, covariate_columns, treatment_columns, target_columns)
	Xtr, logYtr, Xval, logYval, Xte, logYte = dataloader.preprocessData(train_frac=TrainValTest_split[0], val_frac=TrainValTest_split[1], test_frac=TrainValTest_split[2])
	
	data_preprocessed_time = time.time()
	print("Time to preprocess:", data_preprocessed_time - start_time)

	print("Train Size:", Xtr.shape)
	print("Val Size:",	 Xval.shape)
	print("Test Size:",  Xte.shape)
	treereg = TreeRegression(rf_n_estimators, rf_max_depth)

	model_filename = outdir + "RF_postgres_Nest{}_maxD{}_{}.pk".format(rf_n_estimators, rf_max_depth, current_target)
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
	evaluationTimestamp = time.time()
	print("Evaluation time:", evaluationTimestamp - loaded_model_timestamp)
	
	ti_preds, ti_biases, ti_contribs, ti_runtime = compute_ti_attribution(rf, Xte[:500])
	shap_values, shap_runtime = compute_shap_attribution(rf, Xte[:500])
	print(ti_contribs.shape)
	print(shap_values.shape)
	utils.print_spearmanr(ti_contribs, shap_values)
	
	##### Write the comparison data to a file
	timing_info.append(data_preprocessed_time - start_time)
	timing_info.append(rf_trained_time - data_preprocessed_time)
	timing_info.append(loaded_model_timestamp - data_preprocessed_time)
	timing_info.append(evaluationTimestamp - loaded_model_timestamp)
	timing_info.append(ti_values_time - evaluated_time)
	timing_info.append(shap_values_time - ti_values_time)
	utils.write_timing_info_file(timing_info_outfile, timing_info)

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Parser for training RF on dataset')

	parser.add_argument('--save_model', help='Boolean flag to save a model or not')
	parser.add_argument('--evalutaion', help='Boolean flag to evaluate or not')
	parser.add_argument('--dataset_dir', help='Path to dataset csv file')
	parser.add_argument('--current_target', help='string label for current target label')
	parser.add_argument('--num_tree_estimators', help='Number of trees in Foreset based regression model')
	parser.add_argument('--max_depth', help='Maximum Depth each individual tree can go to')
	parser.add_argument('--outdir', help='Output directory to save the model to')
	parser.add_argument('--treatmentTraining', help='Boolean Flag, if true: Train with a configure of treatment variables')
	parser.add_argument('--treatment_combination', help='Configuration of treatment variables')
	parser.add_argument('--timing_info_outfile', help='Output file to store timing information')
	parser.add_argument('--TrainValTest_split', help='Tuple with split ratios of dataset')

	args = vars(parser.parse_args())

	if (args['evalutaion'].lower() == 'true'):
		if (args['treatmentTraining'].lower() == 'false'):
			main((args['save_model'].lower()=='true'), args['outdir'], args['dataset_dir'], args['current_target'], int(args['num_tree_estimators']), int(args['max_depth']), \
								args['timing_info_outfile'], make_tuple(args['TrainValTest_split']) )
		else:
			print(args['treatment_combination'])
			print(make_tuple(args['treatment_combination']))
			mainTreatments((args['save_model'].lower()=='true'), args['outdir'], args['dataset_dir'], args['current_target'], int(args['num_tree_estimators']), int(args['max_depth']), \
								args['timing_info_outfile'], make_tuple(args['treatment_combination']), make_tuple(args['TrainValTest_split']) )
	else:
		trainRF( args['save_model'].lower()=='true', args['outdir'], args['dataset_dir'], args['current_target'], int(args['num_tree_estimators']), int(args['max_depth']),	make_tuple(args['TrainValTest_split']) )

