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
import glob

from postgresql_dataConfig import *

pandas_seed = 482

def getRandomFile(testFileName, trainDir):
	fileName = "covComb(" + re.search('\((.*)\)', testFileName).group(1) + ")_train.csv"
	df = pd.read_csv(os.path.join(trainDir, fileName))
	X_train = df[feature_columns]
	if (X_train.size > 1):
		return X_train.sample(n=1, random_state=pandas_seed);
	else:
		return None
	
def calculateFeatureDifference(baselineContribution, testContribution):
	"""
	Take difference of testContribution dependending on its baseline contribution
	"""
	return testContribution - baselineContribution

def _getRankingFiles(shapFile, tiFile, shapBaseline, tiBaseline):	
	"""
	Get Ranking Files
	"""
	if not os.path.exists(tiFile):
		raise exception('no ti file exists against corresponding shap file')

	#shapvalues = np.loadtxt(shapFile)
	shapvalues = pd.read_csv(shapFile)[feature_columns].values
	shapdifferenceranking = calculateFeatureDifference(shapBaseline, shapvalues)
	print(shapvalues.shape, "//", shapdifferenceranking.shape)

	tivalues = pd.read_csv(tiFile)[feature_columns].values
	tidifferenceranking = calculateFeatureDifference(tiBaseline, tivalues)
	print(tivalues.shape, "//", tidifferenceranking.shape)

	return shapdifferenceranking, tidifferenceranking


def getRankingFiles(modelDir, testDir, trainDir, outDir):
	"""
	Get Ranking Files
	"""
	rf = pk.load(open(modelDir, 'rb'))
	shapDataFiles_All = glob.glob(os.path.join(testDir, 'interpreted*SHAP*test.txt'))
	# tiDataFiles = glob.glob(os.path.join(testDir, '*TI*test.txt'))

	shapContributions = np.array([])
	tiContributions = np.array([])
	explainer = shap.TreeExplainer(rf)

	print("Number of shapDataFiles_All:", len(shapDataFiles_All))
	BaselineCases = []
	tiDataFiles = []
	shapDataFiles = []
	for shapFile_id, shapFile in enumerate(shapDataFiles_All):
		shapDataFiles.append(shapFile)
		tiFile = shapFile
		tiFile = tiFile.replace("SHAP", "TI")
		tiDataFiles.append(tiFile)

		BaselineCase = getRandomFile(shapFile, trainDir)
		if (BaselineCase is not None):
			BaselineCases.append(BaselineCase)
		else:
			shapDataFiles.pop(-1)
			tiDataFiles.pop(-1)

	##### Convert to DataFrame #####
	BaselineCases_df = pd.concat(BaselineCases)
	print("In df:", len(BaselineCases), "/", BaselineCases_df.shape)

	##### Compute Feature Attribution and Get the Runtime for TI and SHAP #####
	import time
	start_time = time.time()
	_, _, tiBaselines = ti.predict(rf, BaselineCases_df)
	ti_time = time.time()
	shapBaselines = explainer.shap_values(BaselineCases_df)
	shap_time = time.time()
	print("Run time on {} cases: TI {}, SHAP {}".format(BaselineCases_df.shape[0], ti_time - start_time, shap_time - ti_time))
	#############################################

	######## Compute Difference Attributions from baseline for each template and every test case ###########
	#shapcontributions = np.array([])
	#ticontributions = np.array([])
	for template_id, (_shapBaseline, _tiBaseline) in enumerate(zip(shapBaselines, tiBaselines)):
		shapContrib, tiContrib = _getRankingFiles(shapDataFiles[template_id], tiDataFiles[template_id], _shapBaseline, _tiBaseline)
		#if shapcontributions.size == 0:
		#	shapcontributions = shapContrib
		#	ticontributions = tiContrib
		#else:
		#	shapcontributions = np.vstack((shapcontributions, shapContrib))
		#	ticontributions = np.vstack((ticontributions, tiContrib))

		########### Write outputs of difference contributions ########
		print(shapDataFiles[template_id])
		shapBaseline_outfile = os.path.join(outDir, os.path.basename(shapDataFiles[template_id]).replace("interpreted", "diffInterpreted").replace("SHAP", "SHAP_baseline_{}".format(template_id)))
		tiBaseline_outfile = os.path.join(outDir, os.path.basename(shapDataFiles[template_id]).replace("interpreted", "diffInterpreted").replace("SHAP", "TI_baseline_{}".format(template_id)))
		np.savetxt(shapBaseline_outfile, BaselineCases_df.iloc[template_id])
		np.savetxt(tiBaseline_outfile, BaselineCases_df.iloc[template_id]) 

		shapDiffContrib_outfile = os.path.join(outDir, os.path.basename(shapDataFiles[template_id]).replace("interpreted", "diffInterpreted"))
		tidiffContrib_outfile = os.path.join(outDir, os.path.basename(tiDataFiles[template_id]).replace("interpreted", "diffInterpreted"))

		pd.DataFrame(shapContrib, columns=feature_columns).to_csv(shapDiffContrib_outfile)
		pd.DataFrame(tiContrib, columns=feature_columns).to_csv(tidiffContrib_outfile)
		#np.savetxt(shapDiffContrib_outfile, shapContrib)
		#np.savetxt(tidiffContrib_outfile, tiContrib)
		print("Saved for {}".format(shapDiffContrib_outfile))


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='Parser for training RF on dataset')
	parser.add_argument('--model_dir', help='Path to trained model to be analysed')
	parser.add_argument('--test_dir', help='Path to testing templates')
	parser.add_argument('--train_dir', help='Path to training templates')
	parser.add_argument('--outdir', help='Path to save the outputs of contribution')

	args = vars(parser.parse_args())
	getRankingFiles(args['model_dir'], args['test_dir'], args['train_dir'], args['outdir'])




