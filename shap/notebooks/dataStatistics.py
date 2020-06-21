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

## Generate distribution histograms for each treatment variables

def getStats(datasetPath, current_target, TrainValTest_split=(1.0,0.0,0.0), outdir=None):
	"""
	Train a Random Forest with given training dataset
	"""
	dataloader = DataLoader(datasetPath, covariate_columns, treatment_columns, target_columns)
	X, logY, _, _, _, _ = dataloader.preprocessData(train_frac=TrainValTest_split[0], val_frac=TrainValTest_split[1], test_frac=TrainValTest_split[2])

	#memory_level = {}
	plot_id = 0
	for treat in ['memory_level', 'page_cost', 'index_level']:
		for _mem_level in [0, 1, 2]:
			#memory_level[_mem_level] = X.index[X['memory_level'] == str(_mem_level)]
			#memory_level[_mem_level] = X.loc[X['memory_level'] == _mem_level]
			logY_selected = logY[X[treat] == _mem_level]
			#print(logY_selected.loc[:, current_target])
			mn = logY_selected.loc[:, current_target].mean()
			array = logY_selected.loc[:, current_target].values
			#mode = np.max(logY_selected.loc[:, current_target])
			hist = np.histogram(array, bins=20)[0]
			#mode = logY_selected.loc[:, current_target].mode(1)
			mode = hist[np.argmax(hist)]
			print(mode)
			median = logY_selected.loc[:, current_target].median()
			

			plt.figure(plot_id)
			plt.title(treat + "=" + str(_mem_level))
			plt.plot(np.linspace(int(min(array)), int(max(array)), 20), hist)
			#plt.hist(logY_selected.loc[:, current_target])
			plt.axvline(x=mn, label='mean', color='r')
			plt.axvline(x=mode, label='mode', color='cyan')
			plt.axvline(x=median, label='median', color='black')
			plt.legend()
			plt.savefig(treat + '_' + str(_mem_level) + '.png')

			plot_id += 1





if __name__=="__main__":
	datasetPath = "/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgres-results.csv"
	getStats(datasetPath, 'runtime')
