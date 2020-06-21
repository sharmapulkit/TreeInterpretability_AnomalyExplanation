#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def print_spearmanr(ranking1, ranking2):
	"""
	Provide the correlation analysis of 2 set of rankings
	ranking1, ranking2: list of size = number of data points
			each element is a ranking of features
	returns pd.Series(rank coefficients)
	Prints the Series analysis of rank correlation values between both the rankings
	(Mean, std, Min, 25%, 50%, 75%, Max)
	"""
	coeffs, ps = [], []
	for r1, r2 in zip(ranking1, ranking2):
		coef, p = spearmanr(r1, r2)
		coeffs.append(coef)
		ps.append(p)
	print(pd.Series(coeffs).describe())
	return pd.Series(coeffs)

def write_timing_info_file(outfile, values):
	"""
	Write timing values to outfile
	values: array of values to be written to desired file
	"""
	labels = ["dataPreprocessing", "modelTraining", "modelTrainEvaluation", "modelTestEvaluation", "TItime", 'SHAPtime']
	with open(outfile, 'w') as f:
		f.write(labels[0])
		for l in labels[1:]:
			f.write('\t')
			f.write(l)
		f.write('\n')
		f.write(str(values[0]))
		for val in values[1:]:
			f.write('\t')
			f.write(str(val))
		f.write('\n')
	return

