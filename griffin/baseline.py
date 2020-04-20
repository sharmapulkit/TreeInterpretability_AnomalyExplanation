#!/usr/bin/env python
"""
Pulls a subset of the training data that represents features for a
"typical job" to use as a baseline in determining reasons for job slowness.
Low and high percentile for selection are set to a default of 45% and 55%.
"""

import os

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

__author__ = "Kristin Lieber, Abhiram Eswaran"


def merge(train_x, train_y):
    """
    Args:
        train_x: Training dataframe.
        train_y: Test dataframe
    Returns: Merged dataframe containing features and labels.
    """
    train = train_x.assign(JobRunTime=train_y.values)
    return train


def baseline_data_selection(
    train_features_dict, train_labels_dict, low_perc=0.45, high_perc=0.55
):
    """
    :param train_features_dict: A dictionary where the key is the job name and the value
                                is a training dataframe.
    :param train_labels_dict: A dictionary where the key is the job name and the value
                                is a pandas series of labels.
    :param low_perc: The lower bound of the percentile.
    :param high_perc: The higher bound of the percentile.
    :return: A dictionary of baselines for each job.
    """

    baselines = {}

    for job_name in train_features_dict.keys():
        df = merge(train_features_dict[job_name], train_labels_dict[job_name])
        low_val = df.JobRunTime.quantile(low_perc)
        high_val = df.JobRunTime.quantile(high_perc)
        baselines[job_name] = df[
            (df["JobRunTime"] > low_val) & (df["JobRunTime"] < high_val)
        ]

    return baselines

def getRandomBaselineContributions(contributions):
    """
    Currently return a random data point from the train set
    """
    return contributions[np.random.choice(contributions.shape[0], 1, replace=False)]
    

