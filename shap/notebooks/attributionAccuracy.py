import numpy as np
import pickle as pk
import pandas as pd
from dataLoader import *
import utils
from sklearn.metrics.pairwise import cosine_similarity

from postgresql_dataConfig import *

Path = "/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/topFeatAccuracy/"
baselinesPath = Path + "_interpreted_SHAP_outs_500_Baseline.txt"
# baselines = np.loadtxt(Path + "_interpreted_SHAP_outs_500_Baseline.txt") #pd.read_csv(Path + "_interpreted_SHAP_outs_500_Baseline.txt").values
#tests = np.loadtxt(Path + "_interpreted_SHAP_outs_500_indexlevel_increased2.txt") #pd.read_csv(Path + "_interpreted_SHAP_outs_500_indexlevel_increased1.txt").values

#baselines = np.loadtxt(Path + "_interpreted_TI_outs_500_Baseline.txt") #pd.read_csv(Path + "_interpreted_SHAP_outs_500_Baseline.txt").values
#tests = np.loadtxt(Path + "_interpreted_TI_outs_500_indexlevel_increased2.txt") #pd.read_csv(Path + "_interpreted_SHAP_outs_500_indexlevel_increased1.txt").values

#baselines = pd.read_csv(baselinesPath, header=None).values
#datapath = Path+'some_500_test_points_Baseline.csv'
#baselinesDL = DataLoader(datapath, covariate_columns, treatment_columns, target_columns)
#_, _, _, _, baselinesD, _ = baselinesDL.preprocessData(0.0, 0, 1.0)
#print(baselinesD.iloc[0])


feat='indexlevel'
attribMethod='TI'
for _trainRatio in range(1, 10, 2):		
		trainRatio = _trainRatio/10
		baselinesPath = Path + "_interpreted_{}_outs_500_trainRatio{}_Baseline.txt".format(attribMethod, trainRatio)
		
		baselines = pd.read_csv(baselinesPath)
		baselines = baselines.drop(columns=['Unnamed: 0'], axis=1)
		baselines = baselines.values

		print("TrainRatio:", trainRatio)
		shaptestsPath = Path + "{}/_interpreted_{}_outs_500_trainRatio{}_{}_increased2.txt".format(feat, attribMethod, trainRatio, feat)
		#if (attribMethod == 'TI'):
		#	shaptestsPath = Path + "{}/_interpreted_{}_outs_500_trainRatio0.'{}'_'{}'_increased2.txt".format(feat, attribMethod, _trainRatio, feat)
		# titests = np.loadtxt(Path + "{}/_interpreted_TI_outs_500_trainRatio'{}'_'{}'_increased2.txt".format(feat, trainRatio, feat))

		#shaptestsDL = DataLoader(shaptestsPath, covariate_columns, treatment_columns, target_columns)
		#_, _, _, _, shaptests, _ = shaptestsDL.preprocessData(0, 0, 1.0)
		shaptests = pd.read_csv(shaptestsPath)
		shaptests = shaptests.drop(columns=['Unnamed: 0'], axis=1)
		shaptests = shaptests.values

		gt_idx = 12
		diff = shaptests - baselines

		tp = 0
		counts = [0]*13
		counts_baseline = counts.copy()
		counts_tests = counts.copy()
		for row_iter, row in enumerate(diff):
			row_sorted = np.argsort(row)

			counts[np.argmax(row)] += 1
			counts_baseline[np.argmax(baselines[row_iter])] += 1
			counts_tests[np.argmax(shaptests[row_iter])] += 1
			if (row_sorted[-1] == gt_idx or row_sorted[-2] == gt_idx):
				tp += 1

		counts = np.array(counts)
		counts_baseline = np.array(counts_baseline)
		counts_tests = np.array(counts_tests)

		# print("counts:", counts)
		# print("counts baseline:", counts_baseline)
		#print("counts tests:", counts_tests)

		#print("diff similarity with baseline:", cosine_similarity(counts[None, :], counts_baseline[None, :]))
		#print("diff similarity with tests:", cosine_similarity(counts[None, :], counts_tests[None, :]))
		#print("similarity between baseline and tests:", cosine_similarity(counts_baseline[None, :], counts_tests[None, :]))

		print("Top 2 Attribution Accuracy:", tp/len(diff))

