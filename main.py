import pandas as pd
import sys
import os
from Preprocessing.Preprocessing import Preprocessor
from FeatureSelection.Features import FeaturesSelector
from Models.Modeling import Models
from Models.ResultsAnalysis import ResultsAnalysis
from Models.Clustering import Clustering

import pickle
import numpy as np
from scipy.stats import spearmanr

from sklearn.preprocessing import LabelBinarizer
from constants import __AXIVITY_FEATURE_NAMES_LIST__, __LABELING_METHOD__, __FEATURE_TYPE_DICT__

__LABELS_COLUMN_NAMES_FOR_METHOD__ = {"isVR":"arm"}



def read_args():

    input_path = sys.argv[2]
    data_file = sys.argv[3]

    label_file = sys.argv[4]

    filter_conditions = sys.argv[5:]
    if type(filter_conditions) == type(str):
        filter_conditions = [filter_conditions]

    return input_path, data_file, label_file, filter_conditions

def read_data(labels_type = __LABELING_METHOD__):

    input_path, data_file, labels_file, filter_conditions = read_args()
    df_X = pd.read_csv(os.path.join(input_path, data_file))
    df_y = pd.read_csv(os.path.join(input_path, labels_file))

    # apply filter on data and labels (e.g. only PD patients, only specific center):
    if filter_conditions != ["all"]:
        print("DATA FILTER APPLIED: " + str(filter_conditions))
        for f in filter_conditions:
            filter_name, filter_value = f.split("_")
            df_y = df_y[df_y[filter_name] == int(filter_value)]

    # Define index:
    df_X.set_index("FileName", inplace=True)
    df_y.set_index("ID", inplace=True)

    # align data with labels (after possivle filtering):
    df_X = df_X[df_X.index.isin(list(df_y.index))]

    # Keep only subject ID and the relevant labels:
    df_y = df_y[[labels_type]]

    execStrDescription = data_file.split('\\')[1].split('.')[0] + "_" + str(filter_conditions) + "_" + __LABELING_METHOD__

    return df_X, df_y, execStrDescription

def prepare_tables(path, dType):

    # read data file and prepare different files: all visits, only first visit, differences between visits
    dataFilesList = os.listdir(os.path.join(path, dType))


def merge_features_table(path):

    fileType = ["delta", "first", "second", "third", "fourth"]
    for fType in fileType:
        sleepFile=pd.read_csv(os.path.join(path, "mergedData", "sleep_data_" + fType + "_visit.csv"))
        axivityFile = pd.read_csv(os.path.join(path, "mergedData", "axivity_data_" + fType + "_visit.csv"))

        all_df = pd.merge(sleepFile, axivityFile, on=["FileName"], how="inner")
        all_df.to_csv(os.path.join(path,"allData", "all_data_" + fType + "_visit.csv"))



def compute_labels(df_labels_data, labeling_method="isVR"):


    return



def prepare_labels(path, fileName, demog, severity):

    return


def main():

    configuration = sys.argv[1]
    input_path = sys.argv[2]

    print("Starting pipeline with labeling by: " + __LABELING_METHOD__)

    if configuration == "preparation":
        prepare_tables(input_path, "sleep")
        prepare_tables(input_path, "axivity")
        merge_features_table(input_path)
        prepare_labels(input_path)

    if configuration == "all" or configuration == "preprocessing":
        X, y, execDesc = read_data()
        Processor = Preprocessor(X,y, input_path)
        X_processed, y_processed = Processor.initial_preprocessing()

    if configuration == "all" or configuration == "clusterModels":
        cls = Clustering(X_processed, y_processed)
        X_enriched = cls.find_clusters_with_GMM()

    if configuration == "all" or configuration == "FeatureSelection":
        Selector = FeaturesSelector(X_enriched, y_processed, input_path)
        Selector.select_features()
        X_filtered = Selector.getFilteredFeatures(forceGMMFeatures=False)
        with open(os.path.join(input_path, "debug_files", execDesc + '.p'), 'wb') as outputFile:
            pickle.dump(list(X_filtered.columns), outputFile)

    if configuration == "all" or configuration == "Modeling":
        print("**************************")
        print("Training models: ")

        print("Number of samples is: " + str(len(X_filtered)))
        print("Number of features is: " + str(len(X_filtered.columns)))
        print("Number of positive samples: " + str((sum(y_processed.values))))
        Model = Models(X_filtered, y_processed)
        scores = []
        for i in range(10):
            scores.append(Model.train_models(seed = i))
        print(["f1","precision","recall/sensitivity","specificity","auc"])
        print(np.mean(scores, axis=0))
        print(np.std(scores, axis=0))

        # Model.cluster()
        # print(scores)

        Model.get_misclassifications()

    if configuration == "Analysis":
        Analyzer = ResultsAnalysis(input_path)
        # Analyzer.analyze_selected_features(sys.argv[3], sys.argv[4])
        # Analyzer.compare_specific_features(sys.argv[2], sys.argv[3], sys.argv[4])
        # Analyzer.analyze_correlations(input_path, sys.argv[3], sys.argv[4])
        # Analyzer.get_falls_distribution(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        Analyzer.check_overlaps(sys.argv[2], sys.argv[3])

if __name__ == '__main__':
    main()