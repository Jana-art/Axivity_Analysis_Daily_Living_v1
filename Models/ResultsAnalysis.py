
import pickle
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns

from collections import defaultdict

from constants import __FEATURE_TYPE_DICT__

class ResultsAnalysis():


    def __init__(self, path, mode = "features"):

        self.path = path
        self.mode = mode


    def analyze_selected_features(self, features_path1, features_path2):

        with open(os.path.join(self.path, features_path1), "rb") as input_file1:
            features1 = pickle.load(input_file1)
        with open(os.path.join(self.path, features_path2), "rb") as input_file2:
            features2 = pickle.load(input_file2)

        print(features1)
        print(features2)

        print(set(features1).intersection(set(features2)))

    def compare_specific_features(self, path, dataF, labelsF):

        features = ['slpAP_30sec_Prc10', 'rngML_30sec_Prc10', 'strRegAP_30sec', 'SparcAP_30sec_Prc10', 'rngML_30sec_Prc90', 'HRv_30sec_Prc90', 'WakeTimeNight', 'AniCVStrideTime_30sec', 'PercentWakeNight', 'strRegAP_30sec_Prc10', 'wdAP_30sec_Prc10', 'rngML_30sec', 'rmsML_30sec_Prc90', 'slpAP_30sec', 'rmsML_30sec', 'strRegAP_30sec_Prc90']
        df = pd.read_csv(os.path.join(path, dataF))
        labels = pd.read_csv(os.path.join(path, labelsF))

        df["ID"] = df["FileName"].apply(lambda x: x[0:6])
        df = df[features + ["ID"]]

        allDF = pd.merge(df, labels, on=["ID"], how="inner")

        pdDf = allDF[allDF["GROUP"] == 3]

        VR = pdDf[pdDf["binaryLabels"] == 1]
        TT = pdDf[pdDf["binaryLabels"] == 0]

        for f in features:
            print(f + " median for VR PD subjects is: " + str(np.median(VR[f])))
            print(f + " median for TT PD subjects is: " + str(np.median(TT[f])))

            plt.hist(VR[f], bins=10, alpha=0.5, label="VR")
            plt.hist(TT[f], bins=10, alpha=0.5, label="TT")
            plt.legend()
            plt.title(f)
            plt.show()


    def analyze_correlations(self, path, dataF, labelsF):

        df = pd.read_csv(os.path.join(path, dataF))
        labels = pd.read_csv(os.path.join(path, labelsF))

        df["ID"] = df["FileName"].apply(lambda x: x[0:6])
        df_frag = df[__FEATURE_TYPE_DICT__["fragmentation"] + ["ID"]]

        df_frag.set_index("ID", inplace=True)
        labels.set_index("ID", inplace=True)

        # labels = labels[labels.index.isin(list(df_frag.index))]

        merged = pd.merge(df_frag, labels, on="ID", how="inner")
        merged = merged[merged["GROUP"] == 3].dropna()

        # for col in __FEATURE_TYPE_DICT__["fragmentation"]:
        #
        #     c1 = merged[col]
        #     c2 = merged["HOEHN"]
        #     corr = spearmanr(c1, c2)
        #
        #     print("pearson corr between: " + col + " & UPDRS3 is: " + str(corr))

        method = "spearman"
        corr_mat = merged.corr(method)

        corr_mat = corr_mat.loc[__FEATURE_TYPE_DICT__["fragmentation"]]
        corr_mat = corr_mat[["UPDRS", "UPDRS3", "HOEHN"]]

        plt.figure(figsize=(15,6))
        heatmap = sns.heatmap(corr_mat, annot=True, cmap = "BrBG")
        heatmap.figure.subplots_adjust(left=0.3)
        heatmap.set_ylim(0,14)
        heatmap.set_title("Correlation between fragmentation features and PD severity scores (" + method + ")", pad = 12)
        # plt.tight_layout()
        # plt.show()

        plt.savefig(os.path.join(path, "debug_files", method + "_corr_frag_heatmap.png"))#, dpi=300, bbox_inches="tight")

        merged.to_csv(os.path.join(path, "debug_files", "framentation_with_labels.csv"))

    def cross_validation(self):

        return


    def plot_roc(self):


        return


    # create scores summarization:
    def summarize_results_output(self):

        return


    def check_overlaps(self, path1, path2):

        df_data = pd.read_csv(os.path.join(path2))[["FileName","StartTime","StopTime"]]
        df_session_times = pd.read_csv(os.path.join(path1))[["Subject","1st Session"]]
        count = 0

        for n in list(df_data["FileName"]):

            id = n.split("_")
            if id[0] != "01" or id[2] != "01":
                continue
            else:
                subject = "01_" + id[1]

                endTime = pd.to_datetime(df_data[df_data["FileName"] == n]["StopTime"], dayfirst=True)
                sessionStartTime = pd.to_datetime(df_session_times[df_session_times["Subject"] == subject]["1st Session"], dayfirst=True)
                if endTime.dt.date.values[0] >  sessionStartTime.dt.date.values[0]:
                    print(subject) #, endTime.dt.date.values[0], sessionStartTime.dt.date.values[0], (endTime.dt.date.values[0] - sessionStartTime.dt.date.values[0]))
                    count += 1
        print(count)


    def get_falls_distribution(self, path1, path2, path3, path4):

        visits = pd.read_csv(path1)
        fall = pd.read_csv(path2)
        demo = pd.read_csv(path3)
        labels = pd.read_csv(path4)

        labels = labels[labels["lifeQualityResponsive"] == 1]["ID"].values

        vData = {}
        fData = {}
        bad_data = []

        for g in visits.groupby("ID"):
            vData[g[0]] = {}
            vData[g[0]][1] = pd.to_datetime(g[1][g[1]["visitid"] == 1]["SVSTDTC"])
            vData[g[0]][2] = pd.to_datetime(g[1][g[1]["visitid"] == 4]["SVSTDTC"])
            vData[g[0]][3] = pd.to_datetime(g[1][g[1]["visitid"] == 5]["SVSTDTC"])
            vData[g[0]][4] = pd.to_datetime(g[1][g[1]["visitid"] == 6]["SVSTDTC"])

            if len(vData[g[0]][4]) == 0:
                bad_data.append(g[0])

            vData[g[0]][5] = vData[g[0]][4] - pd.DateOffset(months=1)
            vData[g[0]][6] = vData[g[0]][4] - pd.DateOffset(months=2)
            vData[g[0]][7] = vData[g[0]][4] - pd.DateOffset(months=3)
            vData[g[0]][8] = vData[g[0]][4] - pd.DateOffset(months=4)


        for g in fall.groupby("ID"):

            if g[0] in bad_data:
                continue

            fData[g[0]] = defaultdict(int)

            for d in pd.to_datetime(g[1]["event_date"]):

                if len(vData[g[0]][2].values) > 0 and d < vData[g[0]][2].values[0]:
                    fData[g[0]][1] += 1

                elif len(vData[g[0]][3].values) > 0 and d < vData[g[0]][3].values[0]:
                    fData[g[0]][2] += 1

                elif len(vData[g[0]][8].values) > 0 and d < vData[g[0]][8].values[0]:
                    fData[g[0]][7] += 1

                elif len(vData[g[0]][7].values) > 0 and d < vData[g[0]][7].values[0]:
                    fData[g[0]][6] += 1

                elif len(vData[g[0]][6].values) > 0 and d < vData[g[0]][6].values[0]:
                    fData[g[0]][5] += 1

                elif len(vData[g[0]][5].values) > 0 and d < vData[g[0]][5].values[0]:
                    fData[g[0]][4] += 1

                elif len(vData[g[0]][4].values) > 0 and d < vData[g[0]][4].values[0]:
                    fData[g[0]][3] += 1


        TT = demo[demo["arm"] == "TT"]["ID"].values
        VR = demo[demo["arm"] == "TT+VR"]["ID"].values

        # TT = list(set(TT).intersection(set(labels)))
        # VR = list(set(VR).intersection(set(labels)))

        first_per = []
        second_per = []
        third_per = []
        last_month_per = []
        two_month_per = []
        three_month_per = []
        four_month_per = []


        for s in TT:
            if s in fData.keys():
                first_per.append(fData[s][1])
                second_per.append(fData[s][2])
                last_month_per.append(fData[s][4])
                two_month_per.append(fData[s][5])
                three_month_per.append(fData[s][6])
                four_month_per.append(fData[s][7])
                third_per.append(fData[s][3])

        print(len(set(TT).intersection(set(fData.keys()))))
        print(np.mean(first_per))
        print(np.mean(second_per))
        print(np.mean(four_month_per))
        print(np.mean(three_month_per))
        print(np.mean(two_month_per))
        print(np.mean(last_month_per))
        print(np.mean(third_per))


        first_per = []
        second_per = []
        last_month_per = []
        third_per = []
        two_month_per = []
        three_month_per = []
        four_month_per = []

        for s in VR:
            if s in fData.keys():
                first_per.append(fData[s][1])
                second_per.append(fData[s][2])
                last_month_per.append(fData[s][4])
                two_month_per.append(fData[s][5])
                three_month_per.append(fData[s][6])
                four_month_per.append(fData[s][7])
                third_per.append(fData[s][3])

        print(len(set(VR).intersection(set(fData.keys()))))
        print(np.mean(first_per))
        print(np.mean(second_per))
        print(np.mean(four_month_per))
        print(np.mean(three_month_per))
        print(np.mean(two_month_per))
        print(np.mean(last_month_per))
        print(np.mean(third_per))


        print(fData)









