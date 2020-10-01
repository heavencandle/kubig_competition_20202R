import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler

class InputData():
    def __init__(self, filename):
        #read data and handle na's
        print("Processing input data...")
        # import data, drop leftmost col
        original = pd.read_csv(filename)
        original = original.drop(columns=['case_no'])
        # change pos->1, neg->0, na to NaN
        original = original.apply(lambda x: x.astype(str).str.strip())
        original['class'] = original['class'].replace(['pos', 'neg'], [1, 0])
        original = original.replace('na', np.NaN)
        # change back to float
        original = original.astype(np.float32)

        # histogram 여부를 기준으로 column 구분
        attrs = original.columns[1:]
        attr_breakdown = {}
        non_histo = []
        histo = {}

        for a in attrs:
            a = a.split("_")
            try: attr_breakdown[a[0]].append(a[1])
            except KeyError: attr_breakdown[a[0]] = [a[1]]
        for attr, bins in attr_breakdown.items():
            if len(bins) == 1: non_histo.append(attr + "_" + bins[0])
            else: histo[attr] = bins

        # 1. histogram이 아닌 column processing : na는 neg와 pos의 mean으로
        for nh in non_histo:
            mask_pos = original['class'].eq(1) & original[nh].notna()
            mask_neg = original['class'].eq(0) & original[nh].notna()

            pos_mean = original.loc[mask_pos, nh].mean()
            neg_mean = original.loc[mask_neg, nh].mean()

            original[nh] = np.where(original['class'].eq(1) & original[nh].isna(), pos_mean, original[nh])
            original[nh] = np.where(original['class'].eq(0) & original[nh].isna(), neg_mean, original[nh])

        #https://kongdols-room.tistory.com/119
        # 2. histrogram columns processing :
        # na 처리 이전, row wise max, min, actvbin만 추출
        for a, bins in histo.items():
            full_bins = [a+"_"+b for b in bins]
            partial = original[['class']+full_bins]
            no_na = partial[partial.notna()]

            #aggregation of bins
            original[a+"_max"] = no_na.iloc[:, 1:].max(axis=1)
            original[a + "_min"] = no_na.iloc[:, 1:].where(no_na>0).min(axis=1)
            original[a + "_actv"] = no_na.iloc[:, 1:].where(no_na>0).count(axis=1)

            # na processing - na with pos
            attrwise_pos_max = original[a + "_max"].where(original['class'].eq(1)).mean(axis=0)
            attrwise_pos_min = original[a + "_min"].where(original['class'].eq(1)).mean(axis=0)
            attrwise_pos_actv = original[a + "_actv"].where(original['class'].eq(1)).mean(axis=0)

            original[a+"_max"] = np.where(original['class'].eq(1) & original[a+"_max"].isna(), attrwise_pos_max, original[a+"_max"])
            original[a+"_min"] = np.where(original['class'].eq(1) & original[a+"_min"].isna(), attrwise_pos_min, original[a+"_min"])
            original[a+"_actv"] = np.where(original['class'].eq(1) & original[a+"_actv"].eq(0), attrwise_pos_actv, original[a+"_actv"])

            # na processing - na with neg
            attrwise_neg_max = original[a + "_max"].where(original['class'].eq(0)).mean(axis=0)
            attrwise_neg_min = original[a + "_min"].where(original['class'].eq(0)).mean(axis=0)
            attrwise_neg_actv = original[a + "_actv"].where(original['class'].eq(0)).mean(axis=0)

            original[a + "_max"] = np.where(original['class'].eq(0) & original[a + "_max"].isna(), attrwise_neg_max, original[a + "_max"])
            original[a + "_min"] = np.where(original['class'].eq(0) & original[a + "_min"].isna(), attrwise_neg_min, original[a + "_min"])
            original[a + "_actv"] = np.where(original['class'].eq(0) & original[a + "_actv"].eq(0), attrwise_neg_actv, original[a + "_actv"])

            original = original.drop(columns=full_bins)

        #split feature and label
        self.feature = original.iloc[:, 1:]
        self.label =  original['class']

        #Standardization
        print("Standardizing data")
        std_scaler = StandardScaler()
        std_scaler.fit(self.feature)
        original_std = std_scaler.transform(self.feature)
        original_std = pd.DataFrame(original_std, columns=self.feature.columns, index=list(self.feature.index.values))
        self.feature = original_std

        print("Processing finished, saving file")
        file_name = "Train_data_processed_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+".csv"
        # self.processed.to_csv(file_name)
        print("Processed File Saved")

    def dataSplit(self, train_portion, val_portion, test_portion):
        # x = self.processed.iloc[:, 1:].to_numpy()
        # y = self.processed['class'].astype(int).to_numpy()

        x = self.feature.to_numpy()
        y = self.label.astype(int).to_numpy()

        shuffle_indices = np.random.permutation(np.arange(len(y)))  # [1,2,3,4...len] ->#[4509,11,1356,4...len]
        shuffled_x = x[shuffle_indices]  # input x를 shuffle 한 후 list를 numpy 자료형으로
        shuffled_y = y[shuffle_indices]  # output (정답) y를 shuffle
        train_index = int(train_portion * float(len(y)))
        test_index = -1 * int(test_portion * float(len(y)))
        self.x_train, self.x_val, self.x_test = shuffled_x[:train_index], shuffled_x[train_index+1:test_index], shuffled_x[test_index:]
        self.y_train, self.y_val, self.y_test = shuffled_y[:train_index], shuffled_y[train_index+1:test_index], shuffled_y[test_index:]

        self.y_train_one_hot = np.eye(2)[self.y_train]  # [9, 8, 0] -> [[0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0]
        # self.y_train_one_hot = np.squeeze(self.y_train_one_hot, axis=1)  # (45000, 10)
        self.y_val_one_hot = np.eye(2)[self.y_val]  # [9, 8, 0] -> [[0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1, 0]
        # self.y_val_one_hot = np.squeeze(self.y_train_one_hot, axis=1)  # (45000, 10)
        self.y_test_one_hot = np.eye(2)[self.y_test]
        # self.y_test_one_hot = np.squeeze(self.y_test_one_hot, axis=1)
        return self.x_train, self.y_train_one_hot, self.x_val, self.y_val_one_hot, self.x_test, self.y_test_one_hot





