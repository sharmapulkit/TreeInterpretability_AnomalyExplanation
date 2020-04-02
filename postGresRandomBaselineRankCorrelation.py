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

curDir = os.getcwd()
modelPath = "rf_postgresql_runtime_200combos.pk"
testPath = os.path.join(curDir,'postgresTemplates','interpretations_2')
trainPath = os.path.join(curDir,'postgresTemplates','Train_subset')
#rf = pk.loads(modelPath)
rf = pk.load(open(modelPath, 'rb'))

import glob
#print(glob.glob(testPath + '\\*.txt'))
#print(glob.glob(os.path.join(testPath, '*SHAP*.txt')))
shapDataFiles = glob.glob(os.path.join(testPath, '*SHAP*test.txt'))
tiDataFiles = glob.glob(os.path.join(testPath, '*TI*test.txt'))


target_columns    = ['local_written_blocks', 'temp_written_blocks', 'shared_hit_blocks', 'temp_read_blocks', 'local_read_blocks', 'runtime', 'shared_read_blocks']
treatment_columns = ['index_level', 'page_cost', 'memory_level']
covariate_columns = ['rows', 'creation_year', 'num_ref_tables', 'num_joins', 'num_group_by', 'queries_by_user', 'length_chars', 'total_ref_rows', 'local_hit_blocks', 'favorite_count']
feature_columns   = covariate_columns.copy()
feature_columns.extend(treatment_columns)

def getRandomBaselineForFile(testFileName):
    fileName = "covComb(" + re.search('\((.*)\)', testFileName).group(1) + ")_train.csv";
    df = pd.read_csv(os.path.join(trainPath, fileName))
    X_train = df[feature_columns]
    return X_train.sample(n=1);
    


def calculateFeatureDifference(baselineContribution, testContribution):
    """
    Take difference of testContribution dependending on its baseline contribution
    """
    return testContribution - baselineContribution

def getRankingFiles(shapDataFiles):
    shapContributions = np.array([])
    tiContributions = np.array([])
    explainer = shap.TreeExplainer(rf)
    
    for shapFile in shapDataFiles:
        tiFile = shapFile
        tiFile = tiFile.replace("SHAP", "TI")
        print(tiFile)
        
        baseline = getRandomBaselineForFile(shapFile);
        print(baseline.shape)
        ti_preds, ti_biases, ti_contribs = ti.predict(rf, baseline)
        
        
        shap_values = explainer.shap_values(baseline)
    
        baselineContributionTI = getRandomBaselineForFile(tiFile)
        shapDifferenceRanking = calculateFeatureDifference(shap_values,np.loadtxt(shapFile))
        tiDifferenceRanking = calculateFeatureDifference(ti_contribs,np.loadtxt(tiFile))
        
        if not os.path.exists(tiFile):
            raise Exception('No TI File exists against corresponding SHAP File')
        
        #print(np.loadtxt(shapFile).shape)
        if shapContributions.size == 0:
            shapContributions = shapDifferenceRanking
            tiContributions = tiDifferenceRanking
        else:
            shapContributions = np.vstack((shapContributions,shapDifferenceRanking))
            tiContributions = np.vstack((tiContributions,tiDifferenceRanking))


    return shapContributions, tiContributions

shapContributions, tiContributions = getRankingFiles(shapDataFiles)

