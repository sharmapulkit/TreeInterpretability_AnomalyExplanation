"""
This function uses tree interpreter to interpret tree structured model to
obtain feature contributions.
"""
from collections import defaultdict
from os import path

import pandas as pd
from treeinterpreter import treeinterpreter as ti
import numpy as np
import yaml

__author__ = "Liqun Shao"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = ""
__maintainer__ = ""
__email__ = ""
__status__ = "Development"


def load_feature_importance(rf_model, data):
    """
    Generate prediction, bias, feature contributions from Tree Interpreter.
    :param rf_model: model.
    :param data: features as input in DataFrame.
    :return: prediction, bias, feature contributions in DataFrames.
    """
    prediction, bias, contributions = ti.predict(rf_model, data)
    return prediction, bias, contributions


def tree_interpreter_instance(
        rf_model, instances, feature_mapping, job_name_template, baselines
):
    """
    Print all the feature contributing scores for each instance.
    :param rf_model: model.
    :param instances: features used to generate importance. Assuming only one instance.
    :param feature_mapping: mappings from features to reasons.
    :param job_name_template: name for job template.
    :param baselines: baseline path in csv file.
    :return: DataFrame each row with 'feature', 'reason', 'importance' and 'trainset_mean'.
    """
    prediction, bias, contributions = ti.predict(rf_model, instances)
    baseline_df = pd.read_csv(baselines)

    # If job name is not in the 6 jobs, return absolute feature importance.
    basepath = path.dirname(__file__)
    config_filepath = path.abspath(path.join(basepath, "../../config.yml"))
    
    # Use job template names from preprocessing in yaml file.
    with open(config_filepath, "r") as config_file:
        try:
            config = yaml.load(config_file)
        except yaml.YAMLError:
            raise Exception("YAML wasn't readable")
    job_templates = config["modeling"]["random_forest"][
        "interested_job_name_templates"]
    non_numeric_features = \
        config["data_acquisition_and_preprocessing"]["data_preprocessing"]["feature_engineering"][
            "one_hot_encoding"]["columns"]

    if job_name_template not in job_templates:
        res_dict = {}
        for i in range(len(instances)):
            res_dict = defaultdict(list)
            print("Instance", i)
            print("Bias (trainset mean)", bias[i])
            print("Predicted runtime", prediction[i])
            print("Feature contributions:")
            for contribution, feature in sorted(
                    zip(contributions[i], list(instances)), key=lambda x: -abs(x[0])
            ):
                res_dict["feature"].append(feature)
                if feature in feature_mapping:
                    res_dict["reason"].append(feature_mapping[feature])
                else:
                    for non_numeric_feature in non_numeric_features:
                        if non_numeric_feature in feature:
                            res_dict["reason"].append(feature_mapping[non_numeric_feature])
                res_dict["importance"].append(round(contribution, 2))
                res_dict["trainset_mean"].append(bias[i])

        # Make all the features with non-numeric feature names.
        for idx, feature in enumerate(res_dict["feature"]):
            for non_numeric_feature in non_numeric_features:
                if non_numeric_feature in feature:
                    res_dict["feature"][idx] = non_numeric_feature
        res_dict_df = pd.DataFrame(res_dict)
        res_dict_df = res_dict_df.groupby(['feature', 'reason']).sum().sort_values(by='importance', ascending=False)
        final_df = pd.DataFrame(columns=['feature', 'reason', 'importance', 'trainset_mean'])
        for index, row in res_dict_df.iterrows():
            final_df.loc[len(final_df)] = index[0], index[1], row['importance'], row['trainset_mean']

        return final_df
    
    #Part that must be of interest to the UMASS Project
    else:
        columns = ["index"]
        columns.extend(instances.columns.values)
        columns.extend(["baseline_runtime", "predicted_baseline", 'job_name'])
        select_df = baseline_df.loc[baseline_df["job_name_template"] == 'JobNameTemplate_' + job_name_template]
        select_df = select_df.reset_index(drop=True)
        select_df.columns = columns
        for i in range(len(instances)):
            res_dict = defaultdict(list)
            for contribution, feature in sorted(
                    zip(contributions[i], list(instances)), key=lambda x: -abs(x[0])
            ):
                res_dict["feature"].append(feature)
                if feature in feature_mapping:
                    res_dict["reason"].append(feature_mapping[feature])
                else:
                    for non_numeric_feature in non_numeric_features:
                        if non_numeric_feature in feature:
                            res_dict["reason"].append(feature_mapping[non_numeric_feature])
                res_dict["importance"].append(
                    round(contribution, 2) - select_df[feature].item()
                )
                res_dict["trainset_mean"].append(bias[i])
                res_dict["baseline_runtime"].append(select_df["baseline_runtime"].item())
                res_dict["predicted_baseline"].append(
                    select_df["predicted_baseline"].item()
                )

        # Make all the features with non-numeric feature names.
        for idx, feature in enumerate(res_dict["feature"]):
            for non_numeric_feature in non_numeric_features:
                if non_numeric_feature in feature:
                    res_dict["feature"][idx] = non_numeric_feature

        # Then group by feature names and sum on feature importance.
        res_dict_df = pd.DataFrame(res_dict)
        res_dict_df = res_dict_df.groupby(['feature', 'reason']).sum()
        final_df = pd.DataFrame(columns=['feature', 'reason', 'importance', 'trainset_mean', 'baseline_runtime',
                                         'predicted_baseline'])
        for index, row in res_dict_df.iterrows():
            final_df.loc[len(final_df)] = index[0], index[1], row['importance'], row['trainset_mean'], row[
                'baseline_runtime'], row['predicted_baseline']

        return final_df.sort_values(by=['importance'], ascending=False)


def tree_interpreter_entire(rf_model, train_x, train_y):
    """
    Plot positive and negative feature contributing figure with tree interpreter.
    Take the mean of all the instances' feature importance.
    :param rf_model: model.
    :param train_x: features.
    :param train_y: runtime.
    :return: feature importance for all the instances.
    """
    prediction, bias, contributions = ti.predict(rf_model, train_x)
    feature_dict = dict((k, []) for k in list(train_x))
    for i in range(len(train_x)):
        for contribution, feature in sorted(
                zip(contributions[i], list(train_x)), key=lambda x: -abs(x[0])
        ):
            feature_dict[feature].append(round(contribution, 2))
    feature_mean_dict = {}
    for feature in train_x.columns:
        feature_mean_dict[feature] = np.mean(feature_dict[feature])
    feature_mean_dict["baseline_runtime"] = np.mean(train_y)
    feature_mean_dict["predicted_baseline"] = np.mean(prediction)

    return feature_mean_dict
