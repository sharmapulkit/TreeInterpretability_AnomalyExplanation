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

from dataLoader import *
from treeRegressor import *
import utils

epsilon = 0.00001

####### Define Output, Covariate and Treatment columns names
target_columns 	  = ['local_written_blocks', 'temp_written_blocks', 'shared_hit_blocks', 'temp_read_blocks', 'local_read_blocks', 'runtime', 'shared_read_blocks']
treatment_columns = ['index_level', 'page_cost', 'memory_level']
covariate_columns = ['rows', 'creation_year', 'num_ref_tables', 'num_joins', 'num_group_by', 'queries_by_user', 'length_chars', 'total_ref_rows', 'local_hit_blocks', 'favorite_count']
feature_columns   = covariate_columns.copy()
feature_columns.extend(treatment_columns)
######################################################


def mainTreatments(SAVE_RF, outdir, datasetPath, current_target, rf_n_estimators, rf_max_depth, timing_info_outfile, tr_comb=None, TrainValTest_split=(0.6,0.2,0.2)):
	start_time = time.time()
	timing_info = []
	
	dataloader = DataLoader(datasetPath + "datasets/postgres-results.csv", covariate_columns, treatment_columns, target_columns)
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
	utils.print_spearmanr(ti_contribs, shap_values)

	timing_info.append(data_preprocessed_time - start_time)
	timing_info.append(rf_trained_time - data_preprocessed_time)
	timing_info.append(loaded_model_timestamp - data_preprocessed_time)
	timing_info.append(evaluationTimestamp - loaded_model_timestamp)
	timing_info.append(ti_values_time - evaluated_time)
	timing_info.append(shap_values_time - ti_values_time)
	utils.write_timing_info_file(timing_info_outfile, timing_info)

def main():
	treereg = TreeRegression()		
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
	parser = argparse.ArgumentParser(description='Parser for training RF on dataset')
	parser.add_argument('--save_model', help='Boolean flag to save a model or not')
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
	print(args['treatment_combination'])
	print(make_tuple(args['treatment_combination']))
	mainTreatments((args['save_model'].lower()=='true'), args['outdir'], args['dataset_dir'], args['current_target'], int(args['num_tree_estimators']), int(args['max_depth']), \
						args['timing_info_outfile'], make_tuple(args['treatment_combination']), make_tuple(args['TrainValTest_split']) )
