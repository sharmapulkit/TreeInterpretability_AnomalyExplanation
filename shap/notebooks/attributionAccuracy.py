import numpy as np
import pickle as pk
import pandas as pd
from dataLoader import *
import utils
from sklearn.metrics.pairwise import cosine_similarity
import glob
import os

from postgresql_dataConfig import *

Path = "/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/topFeatAccuracy/"
#baselinesPath = Path + "_interpreted_SHAP_outs_500_Baseline.txt"
# baselines = np.loadtxt(Path + "_interpreted_SHAP_outs_500_Baseline.txt") #pd.read_csv(Path + "_interpreted_SHAP_outs_500_Baseline.txt").values
#tests = np.loadtxt(Path + "_interpreted_SHAP_outs_500_indexlevel_increased2.txt") #pd.read_csv(Path + "_interpreted_SHAP_outs_500_indexlevel_increased1.txt").values

#baselines = np.loadtxt(Path + "_interpreted_TI_outs_500_Baseline.txt") #pd.read_csv(Path + "_interpreted_SHAP_outs_500_Baseline.txt").values
#tests = np.loadtxt(Path + "_interpreted_TI_outs_500_indexlevel_increased2.txt") #pd.read_csv(Path + "_interpreted_SHAP_outs_500_indexlevel_increased1.txt").values

#baselines = pd.read_csv(baselinesPath, header=None).values
#datapath = Path+'some_500_test_points_Baseline.csv'
#baselinesDL = DataLoader(datapath, covariate_columns, treatment_columns, target_columns)
#_, _, _, _, baselinesD, _ = baselinesDL.preprocessData(0.0, 0, 1.0)
#print(baselinesD.iloc[0])

def main1():
	feat='indexlevel'
	attribMethod='SHAP'
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

		gt_idx = 10
		diff = (shaptests - baselines)

		tp = 0
		counts = [0]*13
		counts_baseline = counts.copy()
		counts_tests = counts.copy()
		epsilon = 0.01
		for row_iter, row in enumerate(diff):
			row_sorted = np.argsort(row) #/np.abs(baselines[row_iter]))

			counts[row_sorted[-1]] += 1
			counts_baseline[np.argmax(baselines[row_iter])] += 1
			counts_tests[np.argmax(shaptests[row_iter])] += 1
			if (row_sorted[-1] == gt_idx): # or row_sorted[-2] == gt_idx):
				tp += 1

		counts = np.array(counts)
		counts_baseline = np.array(counts_baseline)
		counts_tests = np.array(counts_tests)

		# print("counts:", counts)
		# print("counts baseline:", counts_baseline)
		#print("counts tests:", counts_tests)

		print("diff similarity with baseline:", cosine_similarity(counts[None, :], counts_baseline[None, :]))
		print("diff similarity with tests:", cosine_similarity(counts[None, :], counts_tests[None, :]))
		print("similarity between baseline and tests:", cosine_similarity(counts_baseline[None, :], counts_tests[None, :]))

		print("Top 1 Attribution Accuracy_:", tp/len(diff))



def main():
	#tr_var='page_cost'
	tr_vars = ['index_level', 'page_cost', 'memory_level']
	gt_idxs = [10, 11, 12]
	attribMethod='TI'
	for (tr_var, gt_idx) in zip(tr_vars, gt_idxs):
			for _trainRatio in range(5, 6, 2):
				trainRatio = _trainRatio/10

				baselinesPath = Path + "testCases_inc1/baseline"
				counts_over_files = np.array([0]*13)
				total_num_files = 0 + 0.00001
				total_tp = 0
				accs = []
				tpones = []
				tptwos = []
				tpthrees = []
				#/interpreted_{}_outs_500_trainRatio{}_Baseline.txt".format(attribMethod, trainRatio)
				for baselinefile in glob.glob(baselinesPath+"/*"+str(trainRatio)+"*.txt"):
					if (attribMethod in baselinefile):
						baselinefilesplit = baselinefile.split('_trainRatio')
						intervened_file = baselinefilesplit[0].replace('baseline', tr_var) + '_' + tr_var + '_increased_1_'+ tr_var +'_trainRatio0.'+str(_trainRatio)+'_increased1.txt'
						intervened = pd.read_csv(intervened_file)
						#intervened = intervened.drop(columns=['Unnamed: 0'], axis=1)
						intervened = intervened.values
					
						baselines = pd.read_csv(baselinefile)
						#baselines = baselines.drop(columns=['Unnamed: 0'], axis=1)
						baselines = baselines.values
						
						np.random.seed(2)
						randomBaseline = baselines[np.random.randint(len(baselines))]

						#diff = (intervened - baselines)
						diff = intervened - randomBaseline
						diff = np.abs(diff)

						#diff_file = Path + '../diffInterpretations_templates_randomBaselines_shuffleFixed/diffInterpreted_{}_outs_mb_{}'.format(attribMethod)
						#fname = os.path.basename(baselinefile).replace('interpreted', 'diffInterpreted').split('_trainRatio')[0] + '.txt'
						#diff_file = Path + '../postgresTemplates/diffInterpretations_templates_randomBaseline_shuffleFixed/' + fname
						#if not (os.path.exists(diff_file)):
						#	print("Diff file not found")
						#	continue
						#diff = pd.read_csv(diff_file)
						#diff = diff.drop(columns=['Unnamed: 0'], axis=1)
						##print(diff.columns)
						#diff = np.abs(diff.values)
						#print(diff.shape)
						
						tp = 0
						tpone = 0
						tptwo = 0
						tpthree = 0
						counts = [0]*13
						counts_baseline = counts.copy()
						counts_tests = counts.copy()
						epsilon = 0.01
						for row_iter, row in enumerate(diff):
							row_sorted = np.argsort(row) #/np.abs(baselines[row_iter]))

							counts[row_sorted[-1]] += 1
							counts_baseline[np.argmax(baselines[row_iter])] += 1
							#counts_tests[np.argmax(shaptests[row_iter])] += 1
							if (row_sorted[-1] == gt_idx): # or row_sorted[-2] == gt_idx):
								tpone += 1
								tp += 1
							if (row_sorted[-1] == gt_idx or row_sorted[-2] == gt_idx):
								tptwo += 1
							if (row_sorted[-1] == gt_idx or row_sorted[-2] == gt_idx or row_sorted[-3] == gt_idx):
								tpthree += 1

						counts = np.array(counts)
						counts_baseline = np.array(counts_baseline)
						counts_tests = np.array(counts_tests)
						
						tpones.append(tpone)
						tptwos.append(tptwo)
						tpthrees.append(tpthree)
						total_num_files += len(diff)
						total_tp += tpone
						accs.append(tp/len(diff))

						counts_over_files += counts

				#print("Top 1 Attr. Accuracy; TrainRatio: " + str(trainRatio) + ":", total_tp/total_num_files, " tr:", tr_var)
				print("Top 1 Attr. Accuracy; TrainRatio: " + str(trainRatio) + ":", np.sum(tpones)/total_num_files, " tr:", tr_var)
				#print("Top 2 Attr. Acc; TrainRatio: " + str(trainRatio) + ":", np.sum(tptwos)/total_num_files, "tr:", tr_var)
				#print("Top 3 Attr. Acc; TrainRatio: " + str(trainRatio) + ":", np.sum(tpthrees)/total_num_files, "tr:", tr_var)
				# print(np.mean(pd.Series(accs)))
				print(counts_over_files)


def main_special():
	#tr_var='page_cost'
	tr_vars = ['index_level', 'page_cost', 'memory_level']
	gt_idxs = [10, 11, 12]
	attribMethod='SHAP'
	for (tr_var, gt_idx) in zip(tr_vars, gt_idxs):
			for _trainRatio in range(5, 6, 2):
				trainRatio = _trainRatio/10

				baselinesPath = Path + "testCases/baseline"
				dataPath = Path + "testCases/"+tr_var+"/testData/"
				counts_over_files = []
				total_num_files = 0 + 0.00001
				total_tp = 0
				accs = []
				tpones = []
				tptwos = []
				tpthrees = []
				#/interpreted_{}_outs_500_trainRatio{}_Baseline.txt".format(attribMethod, trainRatio)
				for baselinefile in glob.glob(baselinesPath+"/*"+str(trainRatio)+"*.txt"):
					if (attribMethod in baselinefile):
						baselinefilesplit = baselinefile.split('_trainRatio')
						intervened_file = baselinefilesplit[0].replace('baseline', tr_var) + tr_var + '_increased_2_'+ tr_var +'_trainRatio0.'+str(_trainRatio)+'_increased2.txt'
						intervened = pd.read_csv(intervened_file)
						intervened = intervened.drop(columns=['Unnamed: 0'], axis=1)
						intervened = intervened.values
					
						baselines = pd.read_csv(baselinefile)
						baselines = baselines.drop(columns=['Unnamed: 0'], axis=1)
						baselines = baselines.values
						
						np.random.seed(42)
						randomBaseline = baselines[np.random.randint(len(baselines))]

						#diff = (intervened - baselines)
						diff = intervened - randomBaseline
						print(randomBaseline.shape)
						diff = np.abs(diff)

						testDataPointsfile = dataPath + 'covComb' + baselinefilesplit[0].split('outs_mb')[1] + tr_var + '_increased_2.csv'
						testDataPoints = pd.read_csv(testDataPointsfile)
						testDataPoints = testDataPoints.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
						
						
						tp = 0
						tpone = 0
						tptwo = 0
						tpthree = 0
						counts = [0]*13
						counts_baseline = counts.copy()
						counts_tests = counts.copy()
						epsilon = 0.01
						for row_iter, row in enumerate(diff):
							if (testDataPoints.iloc[row_iter][tr_var] == 2): 
									row_sorted = np.argsort(row) #/np.abs(baselines[row_iter]))

									counts[row_sorted[-1]] += 1
									counts_baseline[np.argmax(baselines[row_iter])] += 1
									#counts_tests[np.argmax(shaptests[row_iter])] += 1
									if (row_sorted[-1] == gt_idx): # or row_sorted[-2] == gt_idx):
										tpone += 1
										tp += 1
									if (row_sorted[-1] == gt_idx or row_sorted[-2] == gt_idx):
										tptwo += 1
									if (row_sorted[-1] == gt_idx or row_sorted[-2] == gt_idx or row_sorted[-3] == gt_idx):
										tpthree += 1
									total_num_files += 1

						counts = np.array(counts)
						counts_baseline = np.array(counts_baseline)
						counts_tests = np.array(counts_tests)
						
						tpones.append(tpone)
						tptwos.append(tptwo)
						tpthrees.append(tpthree)
						#total_num_files += len(diff)
						total_tp += tpone
						accs.append(tp/len(diff))

						counts_over_files.append(counts)

				#print("Top 1 Attr. Accuracy; TrainRatio: " + str(trainRatio) + ":", total_tp/total_num_files, " tr:", tr_var)
				print("Top 1 Attr. Accuracy; TrainRatio: " + str(trainRatio) + ":", np.sum(tpones)/total_num_files, " tr:", tr_var)
				print("Top 2 Attr. Acc; TrainRatio: " + str(trainRatio) + ":", np.sum(tptwos)/total_num_files, "tr:", tr_var)
				print("Top 3 Attr. Acc; TrainRatio: " + str(trainRatio) + ":", np.sum(tpthrees)/total_num_files, "tr:", tr_var)
				# print(np.mean(pd.Series(accs)))


def main_attrib_accuracy():
	feat='index_level'
	attribMethod='SHAP'
	
	Path = '/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/interpretations/'
	baselinesPath = Path + "../interpreted_{}_allTest.csv".format(attribMethod)
	
	baselines = pd.read_csv(baselinesPath)
	#baselines = baselines.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
	baselines = baselines.values
	
	shaptestsPath = Path + "../interpretations_intervened/interpreted_{}_test_{}.csv".format(attribMethod, feat)
	#shaptestsPath = Path + ""
	#if (attribMethod == 'TI'):
	#	shaptestsPath = Path + "{}/_interpreted_{}_outs_500_trainRatio0.'{}'_'{}'_increased2.txt".format(feat, attribMethod, _trainRatio, feat)
	# titests = np.loadtxt(Path + "{}/_interpreted_TI_outs_500_trainRatio'{}'_'{}'_increased2.txt".format(feat, trainRatio, feat))
	
	#shaptestsDL = DataLoader(shaptestsPath, covariate_columns, treatment_columns, target_columns)
	#_, _, _, _, shaptests, _ = shaptestsDL.preprocessData(0, 0, 1.0)
	shaptests = pd.read_csv(shaptestsPath)
	#shaptests = shaptests.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
	shaptests = shaptests.values
	
	gt_idx = 10
	diff = (shaptests - baselines)
	
	tp = 0
	counts = [0]*13
	counts_baseline = counts.copy()
	counts_tests = counts.copy()
	epsilon = 0.01
	for row_iter, row in enumerate(diff):
		row_sorted = np.argsort(row) #/np.abs(baselines[row_iter]))
	
		counts[row_sorted[-1]] += 1
		counts_baseline[np.argmax(baselines[row_iter])] += 1
		counts_tests[np.argmax(shaptests[row_iter])] += 1
		if (row_sorted[-1] == gt_idx): # or row_sorted[-2] == gt_idx):
			tp += 1
	
	counts = np.array(counts)
	counts_baseline = np.array(counts_baseline)
	counts_tests = np.array(counts_tests)
	
	# print("counts:", counts)
	# print("counts baseline:", counts_baseline)
	#print("counts tests:", counts_tests)
	
	print("diff similarity with baseline:", cosine_similarity(counts[None, :], counts_baseline[None, :]))
	print("diff similarity with tests:", cosine_similarity(counts[None, :], counts_tests[None, :]))
	print("similarity between baseline and tests:", cosine_similarity(counts_baseline[None, :], counts_tests[None, :]))
	
	print("Top 1 Attribution Accuracy_:", tp/len(diff))


def main_attrib_accuracy_all():
	feats=['index_level', 'page_cost', 'memory_level']
	gt_indices = {feats[0]:10, feats[1]:11, feats[2]:12}
	attribMethods=['SHAP', 'TI']
	
	Path = '/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Intervened/explicit_intervention/'
	for feat in feats:
		for attribMethod in attribMethods:
			gt_idx = gt_indices[feat]
			tp = 0
			counts = [0]*13
			counts_baseline = counts.copy()
			counts_tests = counts.copy()
			epsilon = 0.01
			total = 0
			for filename in glob.glob(os.path.join(Path, 'baseline', 'explanations', 'interpreted_{}*'.format(attribMethod))):
				filename = os.path.basename(filename)
				baselinesPath = os.path.join(Path, 'baseline', 'explanations', filename)
				shaptestsPath = os.path.join(Path, feat, "explanations", filename)
				
				baselines = pd.read_csv(baselinesPath)
				baselines = baselines.values
				
				shaptests = pd.read_csv(shaptestsPath)
				shaptests = shaptests.values
				
				diff = (shaptests - baselines)
				total += len(diff)
				
				for row_iter, row in enumerate(diff):
					row_sorted = np.argsort(row) #/np.abs(baselines[row_iter]))
					counts[row_sorted[-1]] += 1
					counts_baseline[np.argmax(baselines[row_iter])] += 1
					counts_tests[np.argmax(shaptests[row_iter])] += 1
					if (row_sorted[-1] == gt_idx): # or row_sorted[-2] == gt_idx):
						tp += 1
				
			counts = np.array(counts)
			counts_baseline = np.array(counts_baseline)
			counts_tests = np.array(counts_tests)
			
			print("Att Acc {}: {}:".format(feat, attribMethod), tp/total)

if __name__=="__main__":
	main_attrib_accuracy_all()
	# main1()
