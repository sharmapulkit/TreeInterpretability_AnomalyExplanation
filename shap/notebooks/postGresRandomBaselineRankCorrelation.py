import numpy as np
import os
from utils import *
import re
import pandas as pd
import pickle as pk
from sklearn.ensemble import RandomForestRegressor
from treeinterpreter import treeinterpreter as ti
from sklearn import preprocessing
import shap
import argparse

from postgresql_dataConfig import *

curDir = os.getcwd()
modelPath = "rf_postgresql_runtime_200combos.pk"
testPath = os.path.join(curDir,'postgresTemplates','interpretations_2')
trainPath = os.path.join(curDir,'postgresTemplates','Train_subset')

import glob

def getRandomFile(testFileName, trainDir):
	fileName = "covComb(" + re.search('\((.*)\)', testFileName).group(1) + ")_train.csv"
	df = pd.read_csv(os.path.join(trainDir, fileName))
	X_train = df[feature_columns]
	if (X_train.size > 1):
		return X_train.sample(n=1);
	else:
		return None
	
def calculateFeatureDifference(baselineContribution, testContribution):
	"""
	Take difference of testContribution dependending on its baseline contribution
	"""
	return testContribution - baselineContribution

def _getRankingFiles(shapFile, tiFile, shapBaseline, tiBaseline):	
	if not os.path.exists(tiFile):
		raise exception('no ti file exists against corresponding shap file')
	
	shapdifferenceranking = calculateFeatureDifference(shapBaseline, np.loadtxt(shapFile))
	tidifferenceranking = calculateFeatureDifference(tiBaseline, np.loadtxt(tiFile))

	return shapdifferenceranking, tidifferenceranking


def getRankingFiles(modelDir, testDir, trainDir, outDir):
	rf = pk.load(open(modelDir, 'rb'))
	shapDataFiles = glob.glob(os.path.join(testDir, '*SHAP*test.txt'))
	# tiDataFiles = glob.glob(os.path.join(testDir, '*TI*test.txt'))

	shapContributions = np.array([])
	tiContributions = np.array([])
	explainer = shap.TreeExplainer(rf)

	print("Number of shapDataFiles:", len(shapDataFiles))
	BaselineCases = []
	tiDataFiles = []
	for shapFile_id, shapFile in enumerate(shapDataFiles):
		tiFile = shapFile
		tiFile = tiFile.replace("SHAP", "TI")
		tiDataFiles.append(tiFile)

		BaselineCase = getRandomFile(shapFile, trainDir)
		if (BaselineCase is not None):
			BaselineCases.append(BaselineCase)
		else:
			shapDataFiles.remove(shapFile)
			tiDataFiles.pop(-1)
	
	# BaselineCases_df = BaselineCases[0]
	#for elem in BaselineCases[1:]:
	#	BaselineCases_df.append(elem, ignore_index=True)
	BaselineCases_df = pd.concat(BaselineCases)
	print("In df:", len(BaselineCases), "/", BaselineCases_df.shape)

	import time
	start_time = time.time()
	_, _, tiBaselines = ti.predict(rf, BaselineCases_df)
	ti_time = time.time()
	shapBaselines = explainer.shap_values(BaselineCases_df)
	shap_time = time.time()
	print("Run time on {} cases: SHAP {}, TI {}".format(BaselineCases_df.shape[0], ti_time - start_time, shap_time - ti_time))

	shapcontributions = np.array([])
	ticontributions = np.array([])
	for template_id, (_shapBaseline, _tiBaseline) in enumerate(zip(shapBaselines, tiBaselines)):
		shapContrib, tiContrib = _getRankingFiles(shapDataFiles[template_id], tiDataFiles[template_id], _shapBaseline, _tiBaseline)
		if shapcontributions.size == 0:
			shapcontributions = shapContrib
			ticontributions = tiContrib
		else:
			shapcontributions = np.vstack((shapcontributions, shapContrib))
			ticontributions = np.vstack((ticontributions, tiContrib))

			########### Write outputs of difference contributions ########
			shapDiffContrib_outfile = os.path.join(outDir, os.path.basename(shapDataFiles[template_id]).replace("interpreted", "diffInterpreted"))
			tidiffContrib_outfile = os.path.join(outDir, os.path.basename(tiDataFiles[template_id]).replace("interpreted", "diffInterpreted"))
			np.savetxt(shapDiffContrib_outfile, shapcontributions)
			np.savetxt(tidiffContrib_outfile, ticontributions)
			print("Saved for {}".format(shapDiffContrib_outfile))


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Parser for training RF on dataset')
	parser.add_argument('--model_dir', help='Path to trained model to be analysed')
	parser.add_argument('--test_dir', help='Path to testing templates')
	parser.add_argument('--train_dir', help='Path to training templates')
	parser.add_argument('--outdir', help='Path to save the outputs of contribution')

	args = vars(parser.parse_args())
	getRankingFiles(args['model_dir'], args['test_dir'], args['train_dir'], args['outdir'])




