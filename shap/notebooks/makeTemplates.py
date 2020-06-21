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

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import itertools
from sklearn.utils import shuffle
from scipy import stats

from postgresql_dataConfig import *

shuffleSeed = 1


PATH='/mnt/nfs/scratch1/s20psharma/TreeInterpretability/dataset/submission/submissionAll/Data'
data = pd.read_csv(os.path.join(PATH, "../../../postgres-results.csv"))

if __name__=="__main__":
    covariate_columns_shortlist = covariate_columns.copy()
    covariate_columns_shortlist.remove('rows')
    covariate_columns_shortlist.remove('length_chars')
    covariate_columns_shortlist.remove('queries_by_user')
    covariate_columns_shortlist.remove('total_ref_rows')
    print(covariate_columns_shortlist)

    ### Get a dataframe of all unique combinations of covariates
    covariate_shortlist_unique_combs = data.groupby([x for x in covariate_columns_shortlist]).size().reset_index().rename(columns={0:'count_covariate'})

    data_covCombs_shortlist = {}
    for cov_comb in covariate_shortlist_unique_combs.values:
        data_subj = data.copy()
        for c_id, c in enumerate(covariate_columns_shortlist):
            data_subj = data_subj[ data.loc[:, c] == cov_comb[c_id] ]
        data_covCombs_shortlist[tuple(cov_comb)] = data_subj


    outDir = os.path.join(PATH, 'Templates')
    for comb_id, cov_combination in enumerate(data_covCombs_shortlist):
        outfile = os.path.join(outDir, 'covComb' + str(cov_combination).replace(" ", "") + '.csv')
        data_covCombs_shortlist[ cov_combination ].to_csv(outfile, index=False)


