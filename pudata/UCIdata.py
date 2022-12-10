import logging
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from sklearn.utils import shuffle
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


class BenchmarkUCIdata:
    def __init__(self, filepath=None, label_pos=0, r_sep=',', min_count=0, hdr=None):
        """
        filepath = filename with its path
        label_pos = column index of the label in the file
        r_sep = record separator in the file e.g. , or ; or '\t'
        min_count = if some class has fewer records, you can drop that class.
        hdr = does file have a header
        """
        self.filepath = filepath
        self.label_pos = label_pos
        self.r_sep = r_sep
        self.min_count = min_count
        self.hdr = hdr

    def transform_labels(self, data):
        """
        If labels are not numeric, convert them to numeric
        """
        y = data[data.columns[self.label_pos]].to_numpy()
        labels = np.unique(y)
        if not isinstance(labels[0], int):  # if one label is char, array will convert all labels to char
            logging.info("labels are not numeric: {0}".format(labels))
            for i in range(len(labels)):
                y[y == labels[i]] = i
        return y

    def drop_tiny_classes(self, data, label):
        """
        If some classes have smaller number of records, drop those classes.
        """
        idx = []
        for v in np.unique(label):
            ix = np.where(label == v)[0]
            if len(ix) > self.min_count:
                idx.extend(ix)
        return data[idx], label[idx]

    def extract_data_labels(self):
        """
        This function selects data and labels from the given input file
        """
        df = pd.read_csv(self.filepath, header=self.hdr, sep=self.r_sep)
        df = df.fillna(0)  # replace NaN/NA with 0

        # mushroom data has alphabets as feature values, convert them to numeric
        if "mushroom" in self.filepath.lower():
            for c in df.columns:
                df[c] = df[c].apply(ord) % 97

        # select data and labels from the file
        X = df.drop(df.columns[self.label_pos], axis=1).to_numpy()
        y = self.transform_labels(df)
        if self.min_count > 0:
            return self.drop_tiny_classes(X, y)
        else:
            return X, y

    def generate_scar_pu_data(self, fraction):
        """
        Generate PU dataset that satisfies the SCAR assumption.
        Flip labels of some fraction of class 1 records from 1 to 0
        """
        data, label = self.extract_data_labels()
        orig_label = deepcopy(label)  # save the original labels

        if len(np.unique(label)) > 2:
            logging.error("more than two classes in the data. use non_scar function!!! PROGRAM WILL HALT")
            exit(-1)
        else:
            # print the record count for each class
            for c in np.unique(label):
                logging.info("class {0} records: {1}".format(c, np.sum(label == c)))

            # flip labels from 1 to 0
            how_many_to_change = round(np.sum(label == 1) * fraction)
            logging.info("changing labels of {0} records from 1 to 0".format(how_many_to_change))
            idx1 = list(np.where(label == 1)[0])
            random_idx = random.sample(idx1, how_many_to_change)
            label[random_idx] = 0
        return data, label, orig_label

    def generate_non_scar_pu_data(self, fraction, rseed=1007):
        """
        Generate PU dataset that does not satisfy the SCAR assumption
        """
        data, label = self.extract_data_labels()

        if len(np.unique(label)) <= 2:
            logging.error("less than three classes in the data. use scar function!!! PROGRAM WILL HALT")
            exit(-1)
        else:
            # generate list of indices and record count for each class
            label_counts = []   # count of each class
            idx_list = []   # list of "list of indices"
            random.seed(rseed)
            unique_labels = sorted(np.unique(label))
            for v in unique_labels:
                idx = np.where(label == v)[0]
                random.shuffle(idx)
                idx_list.append(idx)
                label_counts.append(len(idx))
                logging.info("class {0} records: {1}".format(v, len(idx)))

            # check which class has the highest count and call it unlabeled
            sorted_count_indx = np.argsort(label_counts)
            logging.info("sorted indices: {0}".format(sorted_count_indx))
            p_count_in_u = math.ceil(label_counts[sorted_count_indx[-1]] * fraction / (1 - fraction))
            logging.info("number of positives in the unlabeled set: {0}".format(p_count_in_u))

            # fraction of different positives in unlabeled
            pos_type = len(unique_labels) - 1
            p_frac_in_u = [2 ** i / (2 ** pos_type - 1) for i in range(pos_type)]
            logging.info("different positive fractions in the unlabeled set: {0}".format(p_frac_in_u))

            # PU data indexes
            idx0 = idx_list[sorted_count_indx[-1]]
            idx1 = []
            j = 0
            for i in sorted_count_indx[:-1]:
                v = int(round(p_count_in_u * p_frac_in_u[j]))
                logging.info("add {0} records to unlabeled set from class {1}".format(v, i))
                idx0 = np.concatenate([idx0, idx_list[i][:v]])
                idx1.extend(idx_list[i][v:])
                j += 1
            logging.info("total records in positive and unlabeled set: {0}, {1}".format(len(idx1), len(idx0)))

            idx = np.concatenate([idx1, idx0])
            y_ml = np.concatenate([[1]*len(idx1), [0]*len(idx0)])
            idx, y_ml = shuffle(idx, y_ml, random_state=rseed)

        return X[idx], y_ml, label[idx]


