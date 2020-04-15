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

### Define Output, Covariate and Treatment columns names for postgreSQL Dataset
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

def main(datasetPath, modelPath, outdir_ti_contribs, outdir_shap_contribs, dataStartPoint=None, dataEndPoint=None, TVT=(0.7, 0.1, 0.2), timing_outFile=None):
	##### Load dataset ###### 
	dataloader = DataLoader(datasetPath, covariate_columns, treatment_columns, target_columns)
	num_dataPoints, featSize = dataloader.getShape()
	Xtr, logYtr, Xval, logYval, Xte, logYte = dataloader.preprocessData(train_frac=TVT[0], val_frac=TVT[1], test_frac=TVT[2])
	###### select subset of dataset and run TI, SHAP
	rf = pk.load(open(modelPath, 'rb'))

	if (dataEndPoint is not None):
		if (dataEndPoint < len(Xte)):
			Xsub = Xte[dataStartPoint:dataEndPoint]
		else:
			Xsub = Xte[dataStartPoint:]
	else:
		Xsub = Xte

	ti_preds, ti_biases, ti_contribs, ti_runtime = compute_ti_attribution(rf, Xsub)
	shap_values, shap_runtime = compute_shap_attribution(rf, Xsub)
	print(ti_contribs.shape)
	print(shap_values.shape)
	coeffs = utils.print_spearmanr(ti_contribs, shap_values)

	if (timing_outFile is not None):
		with open(timing_outFile, 'w') as f:
			f.write("TI Runtime:" + str(ti_runtime))
			f.write('\n')
			f.write("SHAP Runtime:" + str(shap_runtime))
			f.write('\n')
	
	np.savetxt(outdir_ti_contribs, ti_contribs)
	np.savetxt(outdir_shap_contribs, shap_values)
	
	
if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Parser for training RF on dataset')
	parser.add_argument('--dataset_dir', help='Path to dataset csv file')
	parser.add_argument('--model_dir', help='Path to trained model to be analysed')
	parser.add_argument('--outdir_ti_contribs', help='Output file to save TI contribution values')
	parser.add_argument('--outdir_shap_contribs', help='Output file to save SHAP contrib values')
	parser.add_argument('--datapoint_start', help='Starting index of dataset under analysis')
	parser.add_argument('--datapoint_end', help='Ending index of dataset under analysis')
	parser.add_argument('--TrainValTest_split', help='Tuple with split ratios of dataset')
	parser.add_argument('--TimingOutFile', help='Run time of SHAP and TI')

	args = vars(parser.parse_args())
	# print(args['treatment_combination'])
	# print(make_tuple(args['treatment_combination']))

	if ((args['datapoint_start'] is not None) and (args['datapoint_end'] is not None)):
		main(args['dataset_dir'], args['model_dir'], args['outdir_ti_contribs'], args['outdir_shap_contribs'], int(args['datapoint_start']), int(args['datapoint_end']), make_tuple(args['TrainValTest_split']))
	else:
		main(args['dataset_dir'], args['model_dir'], args['outdir_ti_contribs'], args['outdir_shap_contribs'], TVT=make_tuple(args['TrainValTest_split']), timing_outFile=args['TimingOutFile'])
