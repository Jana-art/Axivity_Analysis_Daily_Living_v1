import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from itertools import compress
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2
from constants import __ALL_FEATURE_NAMES_LIST_WITH_ENGINEERED__, __LOW_VARIANCE_FEATURES_TRES__, __MAX_NUM_OF_FEATURES_AFTER_UNIVARIATE_FILTERING__, __ENGINEERED_FEATURES__, __FRAGMENTATION_FEATURE_NAMES__
import numpy as np
import os
from matplotlib import pyplot as plt
import itertools
import random

__RELEVANT_FEATURES__ = []
__META_FEATURES__ = []

__FEATURES_SUBSETS_NAMES__ = {"Fragmentation":__FRAGMENTATION_FEATURE_NAMES__}



class FeaturesSelector():


    def __init__(self, processedData, labels, input_path):

        self.data = processedData
        self.labels = labels
        self.featureNames = list(set(__ALL_FEATURE_NAMES_LIST_WITH_ENGINEERED__).intersection(set(self.data.columns)))
        self.filteredFeatures = self.featureNames
        self.initial_amount_of_features = len(self.featureNames)
        self.input_path = input_path

    def select_features(self, subset=[]):

        print("******************************************************************")
        print("Starting feature selection. Initial number of features: " + str(self.initial_amount_of_features))

        if(len(subset) > 0):
            self.predefined_subset_selection(subset)

        self.univariate_selection()
        self.pairwise_selection()
        # self.backwards_selection()
        # self.embedded_selection()

        self.get_filtered_data_with_ID()

        print("Finished feature selection. Number of features after filtering is: " + str(len(self.filteredFeatures)))
        print("Remaining features names: ")
        print(self.filteredFeatures)
        print("******************************************************************")

        return

    def get_filtered_data_with_ID(self):

        featureList = self.filteredFeatures # + ["SubjectWithCenterID"]
        filteredDataID = self.data[featureList]
        filteredDataID.to_csv(os.path.join(self.input_path, "debug_files", "filteredData.csv"))


    def getFilteredFeatures(self, forceGMMFeatures = True):
        if(len(self.filteredFeatures) > 0):
            if forceGMMFeatures:
                self.filteredFeatures = self.filteredFeatures + __ENGINEERED_FEATURES__
            return self.data[self.filteredFeatures]
        else:
            raise ValueError

    def univariate_selection(self):

        print(" starting univariate feature selection:")
        self.remove_zero_variance_features()
        self.univariate_relation()
        print(" finished univariate feature selection. Number of features remaining: " + str(len(self.filteredFeatures)))
        return


    def remove_zero_variance_features(self):

        print("     Removing low variance features, using threshold of " + str(__LOW_VARIANCE_FEATURES_TRES__))
        toFilter = []

        for f in self.filteredFeatures:
            featureCol = self.data[f]
            colVar = np.var(featureCol)
            if(colVar <= __LOW_VARIANCE_FEATURES_TRES__):
                toFilter.append(f)

        self.filteredFeatures = list(set(self.filteredFeatures) - set(toFilter))

        print("     Finished removing low variance features. Number of features remaining: " + str(len(self.filteredFeatures)))


    def univariate_relation(self):

        print("     Performing univariate correlation feature selection...")
        feature_score = SelectKBest().fit(self.data[self.filteredFeatures].to_numpy(), np.ravel(self.labels.to_numpy())).scores_
        df_feature_score = pd.DataFrame(pd.Series(feature_score))

        feature_score_dict = {"FeatureNames":self.filteredFeatures, "FeatureScore":df_feature_score.values.transpose()[0]}
        self.featureScores = pd.DataFrame(feature_score_dict)
        plt.hist(self.featureScores["FeatureScore"])
        plt.show()

        # plt.plot(sorted(self.featureScores["FeatureScore"]))
        # plt.show()

        maximal_features_number = min(int(len(self.data) / 2), len(self.filteredFeatures)-1)

        features_score_thres = sorted(df_feature_score.values.transpose()[0], reverse=True)[maximal_features_number]

        self.featureScores = self.featureScores[self.featureScores["FeatureScore"] > features_score_thres]

        self.filteredFeatures = list(self.featureScores["FeatureNames"])

        print("     Finished univariate correlation feature selection. Number of features remaining: " + str(len(self.filteredFeatures)))


    # remove correlated features
    def pairwise_selection(self, threshold: int = 0.75):

        print(" Performing pairwise correlation feature selection...")

        corr_df = self.data[self.filteredFeatures].astype(float).corr(method='pearson')
        corr_df['2nd_large'] = corr_df.apply(lambda row: row.nlargest(2).values[-1], axis=1)
        mask_threshold: np.ndarray = np.where(abs(corr_df.values) > threshold, 1, 0)
        corr_df['depend#'] = mask_threshold.sum(axis=1) - 1
        df_corr_count = pd.DataFrame(np.unique(corr_df['depend#'].values), columns=['depend_level'])
        bincount = np.bincount(corr_df['depend#'].values)
        df_corr_count['count'] = bincount[df_corr_count['depend_level']]
        sorted_corr_df = corr_df.sort_values(by=['depend#', '2nd_large'], ascending=[True, True])
        independent_set = set()
        ls: list = []
        for depend_level in df_corr_count['depend_level']:
            row_feature_indx = sorted_corr_df[sorted_corr_df['depend#'] == depend_level].index
            if depend_level == 0:
                independent_set = independent_set.union(row_feature_indx)
                continue
            for row in row_feature_indx:
                # get the features indices that has correlation greater than threshold with the feature in row
                row_series = sorted_corr_df.loc[row].drop(['depend#', '2nd_large'])
                col_feature_indx = row_series[abs(row_series) > 0.75].index
                corr_set = set(col_feature_indx)
                if independent_set.isdisjoint(corr_set):
                    independent_set.add(row)
        ls.append([*independent_set, ])
        independent_indx_list = list(itertools.chain.from_iterable(ls))
        self.filteredFeatures = independent_indx_list
        # self.filteredFeatures = list(set(self.filteredFeatures) - set(independent_indx_list))
        print(" number of selected features after pairwise feature selection stage: " + str(len(self.filteredFeatures)))


    def backwards_selection(self):

        return


    def embedded_selection(self):

        print(" Perform embedded feature selection using random forest...")

        # # select top features with RF
        # random.seed(1234)
        # rf = RandomForestClassifier(n_estimators=100, random_state=0)
        # rf.fit(self.data[self.filteredFeatures], self.labels)
        # print(rf.feature_importances_)

        # select top features with RF
        random.seed(1234)
        # clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0)
        clf = LogisticRegression(penalty="l2", class_weight="balanced", random_state=0)
        s = SelectFromModel(clf)
        s.fit(self.data[self.filteredFeatures], self.labels)
        selected = list(compress(__ALL_FEATURE_NAMES_LIST_WITH_ENGINEERED__, s.get_support()))

        self.filteredFeatures = selected

        # self.filteredFeatures = list(set(self.filteredFeatures) - set(selected))

        print(" number of selected features after embedded feature selection stage: " + str(len(self.filteredFeatures)))


    def predefined_subset_selection(self, subsetsToInclude):

        featureSet = set()
        for sub in subsetsToInclude:
            featureSet = featureSet.union(set(__FEATURES_SUBSETS_NAMES__[sub]))

        # self.data = self.data[list(featureSet)]
        self.filteredFeatures = list(featureSet)