import numpy as np
import pandas as pd

if __name__=="__main__":
	TIattributionFile = '/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/interpreted_TI_allTest_noheader.csv'
	SHAPattributionFile = '/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/interpreted_SHAP_allTest_noheader.csv'

	TIattributionFile = '/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/interpretations_shuffleFixed/interpreted_TI_outs_mb(2010,0,0,1,0,0,540)_test.txt'
	SHAPattributionFile = '/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/postgresTemplates/interpretations_shuffleFixed/interpreted_SHAP_outs_mb(2010,0,0,1,0,0,540)_test.txt'
	
	#df = pd.read_csv(TIattributionFile, header=None)
	df = pd.read_csv(TIattributionFile)
	df = df.drop(columns=['Unnamed: 0'], axis=1)
	dfmod = df.abs()
	var = dfmod.var(1)
	mean_variance = np.median(var)
	print("Median Var, TI:", mean_variance)
	
	#df = pd.read_csv(SHAPattributionFile, header=None)
	df = pd.read_csv(SHAPattributionFile)
	df = df.drop(columns=['Unnamed: 0'], axis=1)
	dfmod = df.abs()
	var = dfmod.var(1)
	mean_variance = np.median(var)
	print("Median Var, SHAP:", mean_variance)

