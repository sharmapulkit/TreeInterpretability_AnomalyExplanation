### Create a subset from all of the Templates

import numpy as np
import csv
import xgboost
import shap
import pandas as pd
import matplotlib.pylab as plt
import os
import glob
import pickle as pk
import sklearn
import argparse

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import itertools
from sklearn.utils import shuffle
from scipy import stats

from postgresql_dataConfig import *

shuffleSeed = 1

NUMBER_OF_POINTS_THRESHOLD = 600
MIN_NUMBER_OF_POINTS_THRESHOLD = 20
TRAIN_FRAC = 0.7
PATH='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates'
TARGET_PATH='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data/Templates_subset'+str(NUMBER_OF_POINTS_THRESHOLD)+'/'

def split_train_test_templates():
	unsuccessful = 0
	for filename in os.listdir(PATH):
		filename = os.path.basename(filename)
		cov_comb = filename.rstrip(').csv').split('covComb(')[-1]
		cov_comb = [int(x) for x in cov_comb.split(',')]
		num_datapoints = cov_comb[-1]
		if not (os.path.exists(os.path.join(TARGET_PATH, 'train'))):
			os.mkdir(os.path.join(TARGET_PATH, 'train'))
		if not (os.path.exists(os.path.join(TARGET_PATH, 'test'))):
			os.mkdir(os.path.join(TARGET_PATH, 'test'))
	
		if (num_datapoints < NUMBER_OF_POINTS_THRESHOLD):
			df = pd.read_csv(os.path.join(PATH, filename))

			if (df.shape[0] > MIN_NUMBER_OF_POINTS_THRESHOLD):
				df_train = df.loc[:int(TRAIN_FRAC*num_datapoints)]
				df_test = df.loc[int(TRAIN_FRAC*num_datapoints):]
				
				df_train.to_csv(os.path.join(TARGET_PATH, 'train', filename), index=False)
				df_test.to_csv(os.path.join(TARGET_PATH, 'test', filename), index=False)
			else:
				unsuccessful += 1
				print("Too Few Data Points {}".format(unsuccessful))

	print("Copy Successful")

def select_few_point_templates():
	for filename in os.listdir(PATH):
		cov_comb = filename.rstrip(').csv').split('covComb(')[-1]
		cov_comb = [int(x) for x in cov_comb.split(',')]
		num_datapoints = cov_comb[-1]
		if (num_datapoints < NUMBER_OF_POINTS_THRESHOLD):
			if not (os.path.exists(os.path.join(TARGET_PATH, 'templatewise'))):
				os.mkdir(os.path.join(TARGET_PATH, 'templatewise'))
			os.system('cp ' + '"'  + os.path.join(PATH, filename) + '" "' + os.path.join(TARGET_PATH, 'templatewise', filename) + '"' )
	print("Templates with less than {} data points copied Successful".format(NUMBER_OF_POINTS_THRESHOLD))

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--max_num_points_thresh')
	parser.add_argument('--train_frac')
	parser.add_argument('--ippath')
	parser.add_argument('--oppath')
	args = vars(parser.parse_args())

	NUMBER_OF_POINTS_THRESHOLD = int(args['max_num_points_thresh'])
	TRAIN_FRAC = int(args['train_frac'])/100
	PATH = args['ippath']
	TARGET_PATH = args['oppath']

	split_train_test_templates()


