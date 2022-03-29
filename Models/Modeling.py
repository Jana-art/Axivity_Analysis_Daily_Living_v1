from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
# from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix, roc_auc_score, roc_curve
from constants import __MODELS__, __FOLDS_NUM__
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt

import random

class Models():

    def __init__(self, filteredFeaturesData, labels):

        self.X = filteredFeaturesData
        self.y = labels

        # For later - misclassification analysis
        self.FP_dict = defaultdict(int)
        self.FN_dict = defaultdict(int)
        self.TN_dict = defaultdict(int)
        self.TP_dict = defaultdict(int)
        # self.samplesID = samples_ID

    """
    Training models on the data with the following steps:
    1. Split the data to two sets: (A) train/validation set (B) test set.
    2. Apply cross validation to set (A) for model selection
    3. Choose the best algorithm by the maximal f1 score received on KFold cross validation
    4. Apply the chosen algorithm on the test set from step (1) to evaluate its actual performance on new data
    """
    def train_models(self, seed=0, thres = 0.5, model = "RF", verbose = False):

        # if verbose:

        cv = StratifiedKFold(n_splits=__FOLDS_NUM__, shuffle=True, random_state=seed)

        f1_scores = {}
        precision_scores = {}
        recall_scores = {}
        models = {}

        # keep aside a test set to estimate the performance of the chosen model
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, shuffle=True, random_state = seed)

        # Perform model selection using cross validation:
        for model in __MODELS__:

            if verbose:
                print("Results for model " + model)

            clf = self.choose_model(model, seed)

            models[model] = clf

            f1_scores[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1").mean()
            precision_scores[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="precision").mean()
            recall_scores[model] = cross_val_score(clf, X_train, y_train, cv=cv, scoring="recall").mean()

            # Results:
            if verbose:
                print("f1 score: " + str(f1_scores) + "\nrecall: " + str(recall_scores) + "\nprecision:" + str(precision_scores))

        chosen_model = list(f1_scores.keys())[list(f1_scores.values()).index(np.max(list(f1_scores.values())))]

        # test the chosen model on the test set:
        clf_chosen = models[chosen_model]
        clf_chosen.fit(X_train, y_train)
        pred = clf_chosen.predict(X_test)
        chose_model_f1_score = f1_score(y_test, pred)
        chose_model_precision_score = precision_score(y_test, pred)
        chose_model_recall_score = recall_score(y_test, pred)

        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        chose_model_specificity_score = tn / (tn + fp)

        auc_score = -1
        prob = self.get_prob_score(X_test, y_test, chosen_model, clf_chosen)
        auc_score = roc_auc_score(y_test, prob)
        fpr, tpr, thresh = roc_curve(y_test, prob)

        # plt.plot(fpr, tpr)
        # plt.show()


        self.update_misclassifications(y_test, pred)

        if verbose:
            print("Chosen model is: " + chosen_model)
            print("the f1 score of the model on the test set is: " + str(chose_model_f1_score))
            print("the recall score of the model on the test set is: " + str(chose_model_recall_score))
            print("the precision score of the model on the test set is: " + str(chose_model_precision_score))
            print("the specificity score of the model on the test set is: " + str(chose_model_specificity_score))
            print("the auc score of the model on the test set is: " + str(auc_score))

        return [chose_model_f1_score, chose_model_precision_score, chose_model_recall_score, chose_model_specificity_score, auc_score]

    def choose_model(self, model_name, seed):

        clf = None

        if model_name == "RF":
            clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed)
        elif model_name == "SVM":
            clf = svm_linear = SVC(C=1.0, kernel='linear', random_state=seed)
        elif model_name == "NB":
            clf = GaussianNB()
        elif model_name == "LR":
            clf = LogisticRegression(solver="lbfgs", penalty="l2", class_weight="balanced", random_state=seed)

        return clf


    def update_misclassifications(self, y_test, pred):

        results = pd.DataFrame([y_test.index.values, y_test.values, pred]).transpose()
        results.columns = ["SubjectID","y_test","pred"]
        FP = list(results[(results["y_test"] == 0) & (results["pred"] == 1)]["SubjectID"])
        FN = list(results[(results["y_test"] == 1) & (results["pred"] == 0)]["SubjectID"])
        TN = list(results[(results["y_test"] == 0) & (results["pred"] == 0)]["SubjectID"])
        TP = list(results[(results["y_test"] == 1) & (results["pred"] == 1)]["SubjectID"])

        for s in FP:
            self.FP_dict[s] = self.FP_dict[s] + 1
        for j in FN:
            self.FN_dict[j] = self.FN_dict[j] + 1
        for k in TN:
            self.TN_dict[k] = self.TN_dict[k] + 1
        for n in TP:
            self.TP_dict[n] = self.TP_dict[n] + 1

    def get_misclassifications(self):

        print("FP samples: " + str(self.FP_dict))
        print("FN samples: " + str(self.FN_dict))
        print("TP samples: " + str(self.TP_dict))
        print("TN samples: " + str(self.TN_dict))

    def get_prob_score(self, X_test, y_test, chosen_model, clf_chosen):

        if(chosen_model == "SVM"):
            return clf_chosen.decision_function(X_test).transpose()
        else:
            return clf_chosen.predict_proba(X_test)[:,1]
