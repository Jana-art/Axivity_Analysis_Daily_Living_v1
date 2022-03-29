
import pandas as pd
import os
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from constants import __ALL_FEATURE_NAMES_LIST__, __LABELING_METHOD__

__OUTLIERS_FEATURES_PERCENTAGE_FOR_SAMPLE_EXCLUSION__ = 0.15
__NUMBER_OF_DAYS_FOR_MINIMAL_QUALITY__ = 4
__NUMBER_OF_30SEC_BOUTS_FOR_MINIMAL_QUALITY__ = 40



class Preprocessor():


    def __init__(self, X, y, input_path):

        self.data = X.sort_values(by="SubjectID", axis = 0)
        self.labels = y
        self.sparseFeatures = [] # features which have too many missing values
        self.input_path = input_path

    def initial_preprocessing(self):

        # self.manual_remove_samples()

        self.samples_quality_test()
        self.impute_missing_values()
        self.remove_outliers()
        self.feature_normalization()

        self.create_aligned_labels_vector()

        return self.scaledData, self.labels #, self.samples_ID

    def samples_quality_test(self):

        self.verify_minimal_information_features_and_samples()
        # self.check_samples_quality()

    # Measures for minimal quality of data:
    # * minimum number of days in recording for daily activity features
    # * minimum number of bouts for gait quality features
    # *
    def check_samples_quality(self):

        self.data = self.data[self.data["ValidDays12HR"] >= __NUMBER_OF_DAYS_FOR_MINIMAL_QUALITY__]
        self.data = self.data[self.data["NumberofWalkingBouts_30sec"] >= __NUMBER_OF_30SEC_BOUTS_FOR_MINIMAL_QUALITY__]
        return

    # Keep only features with values in at least %p1 of the samples
    # Keep only samples with values in at least  %p2 of the features:
    def verify_minimal_information_features_and_samples(self, feature_min_fraction = 0.8, sample_min_fraction=0.8):

        col_stats = self.data.count(axis=0)
        sample_stats = self.data.count("columns")

        col_stats = col_stats.where(col_stats > feature_min_fraction * len(self.data)).dropna()
        sample_stats = sample_stats[sample_stats > sample_min_fraction * len(self.data.columns)]

        self.data = self.data[col_stats.index.tolist()]

        self.data = self.data.loc[sample_stats.index.tolist()]

    # Remove samples with high percentage of features having values which are considered as outliers:
    def remove_outliers(self):

        print("Removing outliers: ")
        outlier_count = pd.Series(False, index=self.data.index)
        feature_list = set(self.data.columns).intersection(__ALL_FEATURE_NAMES_LIST__)
        for col in feature_list:
            Q1, Q3 = self.data[col].quantile([.25, .75])
            IQR = Q3 - Q1
            minimum = Q1 - 1.5 * IQR
            maximum = Q3 + 1.5 * IQR
            mask = ~(self.data[col].between(minimum, maximum, inclusive=True))
            outlier_count = mask.astype(int) + outlier_count

        num_features = len(self.data.columns)

        self.data["outliers"] = outlier_count

        outliers = self.data[self.data["outliers"] > __OUTLIERS_FEATURES_PERCENTAGE_FOR_SAMPLE_EXCLUSION__ * num_features]["FileName"]
        if len(outliers):
            print("Outliers removed:")
            print(list(outliers.index))

        self.data = self.data[self.data["outliers"] <= __OUTLIERS_FEATURES_PERCENTAGE_FOR_SAMPLE_EXCLUSION__ * num_features]

    def impute_missing_values(self):

        # imp = SimpleImputer(missing_values=np.nan, strategy='median')
        # self.data=self.data.fillna(self.data.median())
        self.data = self.data.dropna()
        # TODO: update sparse features with columns which have too many missing values, and remove those features
        return


    def feature_normalization(self, method = "normalize"):

        features_list = list(set(self.data.columns).intersection(__ALL_FEATURE_NAMES_LIST__))
        ct = ColumnTransformer([("StandardScaler", StandardScaler(), features_list)], remainder='passthrough')
        self.scaledData = ct.fit_transform(self.data)
        meta_features = [x for x in self.data.columns if x not in features_list]
        self.scaledData = pd.DataFrame(self.scaledData, columns = features_list + meta_features, index=self.data.index)


    def create_aligned_labels_vector(self, labelingMethod = __LABELING_METHOD__):

        labels_for_model = self.labels[self.labels.index.isin(list(self.data.index))]
        labels_for_model.to_csv(os.path.join(self.input_path, "debug_files", "labels.csv"))

        self.labels = labels_for_model[labelingMethod]
        # self.samples_ID = labels_for_model["ID"]

        return



