import numpy as np
import pandas as pd
import rbo
import matplotlib.pyplot as plt


def print_rbo(ranking1, ranking2):
    """
    Provide the correlation analysis of 2 set of rankings
    ranking1, ranking2: list of size = number of data points
            each element is a ranking of features
            
    returns pd.Series(rank based overlap)
    Prints the Series analysis of rank correlation values between both the rankings
    (Mean, std, Min, 25%, 50%, 75%, Max, Histogram)
    """
    corr = []
    idx_arr = np.arange(len(ranking1))
    for r1, r2 in zip(ranking1, ranking2):
        # Sorting the rankings in descending order of the absolute attribution values.
        ti_contributions = [attr_value[1] for attr_value in 
                        sorted(zip(r1, idx_arr), key=lambda temp: -abs(temp[0]))]
        shap_contributions = [attr_value[1] for attr_value in 
                          sorted(zip(r2, idx_arr), key=lambda temp: -abs(temp[0]))]
        rank_correlation = rbo.RankingSimilarity(shap_contributions, ti_contributions).rbo()
        corr.append(rank_correlation)
    print(pd.Series(corr).describe())
    plt.hist(corr) 
    return pd.Series(corr)    

def print_rbo_topk(ranking1, ranking2, k):
    """
    Provide the correlation analysis of top-k features from the 2 list of rankings
    ranking1, ranking2: list of size = number of data points
            each element is a ranking of features
            
    returns pd.Series(top-k rank based overlap)
    Prints the Series analysis of rank correlation values between both the rankings
    (Mean, std, Min, 25%, 50%, 75%, Max, Histogram)
    """
    corr = []
    idx_arr = np.arange(len(ranking1))
    for r1, r2 in zip(ranking1, ranking2):
        
        # Sorting the rankings in descending order of the absolute attribution values.
        ti_contributions = [attr_value[1] for attr_value in 
                        sorted(zip(r1, idx_arr), key=lambda temp: -abs(temp[0]))][:k]
        shap_contributions = [attr_value[1] for attr_value in 
                          sorted(zip(r2, idx_arr), key=lambda temp: -abs(temp[0]))][:k]
        rank_correlation = rbo.RankingSimilarity(shap_contributions, ti_contributions).rbo()
        corr.append(rank_correlation)
    print(pd.Series(corr).describe())
    plt.hist(corr) 
    return pd.Series(corr)    

def print_set_overlap_top_k(ranking1, ranking2, k):
    """
    Provide the set overlap of top-k features from the 2 list of rankings
    ranking1, ranking2: list of size = number of data points
            each element is a ranking of features
            
    returns pd.Series(set overlap)
    Prints the Series analysis of rank correlation values between both the rankings
    (Mean, std, Min, 25%, 50%, 75%, Max, Histogram)
    """
    overlap_values = []
    idx_arr = np.arange(len(ranking1))
    for r1, r2 in zip(ranking1, ranking2):
        
        # Sorting the rankings in descending order of the absolute attribution values.
        ti_contributions = set([attr_value[1] for attr_value in 
                        sorted(zip(r1, idx_arr), key=lambda temp: -abs(temp[0]))][:k])
        shap_contributions = set([attr_value[1] for attr_value in 
                          sorted(zip(r2, idx_arr), key=lambda temp: -abs(temp[0]))][:k])
        overlap_len = len(ti_contributions.intersection(shap_contributions))
        overlap = overlap_len/k
        overlap_values.append(overlap)
    print(pd.Series(overlap_values).describe())
    plt.hist(overlap_values) 
    return pd.Series(overlap_values) 

