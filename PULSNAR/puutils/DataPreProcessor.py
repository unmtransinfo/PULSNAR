import numpy as np
import random
import logging
from scipy import sparse


class MLDataPreprocessing:
    def __init__(self, rseed=7):
        self.rseed = rseed

    def generate_pu_dataset(self, xdata, ml_labels, true_labels, rec_ids):
        """
        Select all class 1 records as positive and class 0 as unlabeled

        Parameters
        ----------
        xdata: feature matrix for ML
        ml_labels: list of ML labels
        true_labels: list of true labels (only for known datasets)
        rec_ids: list of records ids

        Returns
        ------
        positive features
        positive true labels
        positive records ids
        unlabeled features
        unlabeled true labels
        unlabeled records ids
        """

        idx1 = np.where(ml_labels == 1)[0]
        idx0 = np.where(ml_labels == 0)[0]
        random.seed(self.rseed)
        random.shuffle(idx1)
        random.shuffle(idx0)
        # logging.info("Number of positive and unlabeled records: {0}, {1}, {2}".format(len(idx1), len(idx0),
        #                                                                              set(idx0).intersection(idx1)))
        return xdata[idx1], ml_labels[idx1], true_labels[idx1], rec_ids[idx1], \
               xdata[idx0], ml_labels[idx0], true_labels[idx0], rec_ids[idx0]

    def generate_balanced_train_data(self, X1, y1_true, rec1, X0, y0_true, rec0, csr=False):
        """
        randomly selected k records from the unlabeled set, k= |pos|.
        create train data using positive data and randomly selected unlabeled data.
        create test data using the remaining unlabeled data.

        Parameters
        ----------
        X1: positive features
        y1_true: positive true labels
        rec1: positive record ids
        X0: unlabeled features
        y0_true: unlabeled true labels
        rec0: unlabeled record ids
        csr: feature matrix in csr format?

        Returns
        -------
        X_train: train feature matrix
        y_train_ml: ML labels of train data
        y_train_true: true labels of train data
        recs_train: records ids of train data
        X_test: test feature matrix
        y_test_ml: ML label of test data
        y_test_true: true label of test data
        recs_test: record ids of test data
        """
        random.seed(self.rseed)
        all_indx = [i for i in range(len(rec0))]
        idx_tr = random.sample(all_indx, len(rec1))
        idx_te = list(set(all_indx).difference(idx_tr))

        # generate train data
        if csr:
            X_train = sparse.vstack([X1, X0[idx_tr]], format='csr')  # merge the data
        else:
            X_train = np.vstack([X1, X0[idx_tr]])  # merge the data
        y_train_ml = np.concatenate([[1]*len(rec1), [0]*len(rec1)])
        y_train_true = np.concatenate([y1_true, y0_true[idx_tr]])
        recs_train = np.concatenate([rec1, rec0[idx_tr]])

        # generate test data
        X_test = X0[idx_te]
        y_test_ml = np.array([0]*len(idx_te))
        y_test_true = y0_true[idx_te]
        recs_test = rec0[idx_te]

        logging.info("Total train records: {0}, Total test records: {1}".format(len(recs_train), len(recs_test)))
        return X_train, y_train_ml, y_train_true, recs_train, X_test, y_test_ml, y_test_true, recs_test
