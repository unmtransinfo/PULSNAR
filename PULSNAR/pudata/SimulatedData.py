import logging
import numpy as np
import random
import traceback
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
from sklearn.preprocessing import KBinsDiscretizer

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


class SklearnSimulatedData:
    def __init__(self, params):
        self.params = params
        self.rseed = params['random_state']
        self.data = None
        self.labels = None
        self.p_types = 1
        # parameters for Binarization
        self.encode = "ordinal"
        self.strategy = "quantile"
        self.n_bins = 5

    def generate_simulated_data(self, n_pos=10000, n_unlab=10000, pf_in_unlab=0.1, n_features=25, n_classes=2,
                                class_sep=0.3, equal_frac_in_pos=True, discrete_features=False, scar=True, rseed=7):
        """
        Use this function to generate simulated data using the given parameters

        Parameters
        ----------
        n_pos: number of positive records
        n_unlab: number of unlabeled records
        pf_in_unlab: fraction of positives in unlabeled
        n_features: number of features
        n_classes: number of classes
        class_sep: classification difficulty level
        equal_frac_in_pos: equal fraction of different types of positives in the positive set?
        discrete_features: feature value continuous or binary?
        scar: scar data or non_scar data?
        rseed: random seed

        Returns
        ------
        X: simulated data containing positives and unlabeled
        y_ml: ML labels of the data
        y_true: true labels of the data
        """
        self.params['n_samples'] = max(n_pos, n_unlab) * n_classes * 2
        self.params['n_features'] = n_features
        self.params['n_classes'] = n_classes
        self.params['class_sep'] = class_sep
        self.params['random_state'] = rseed
        self.params['n_informative'] = n_features - self.params['n_redundant'] - self.params['n_repeated']
        self.rseed = rseed

        # self.params['n_informative'] = n_classes
        #if scar:
        #    self.params['n_clusters_per_class'] = 2  # use default value for the SCAR data
        #else:
        #    self.params['n_clusters_per_class'] = 1  # set it to 1

        # basic checks before processing
        if scar and n_classes != 2:
            traceback.print_stack()
            logging.error("number of classes should be 2 for SCAR data")
            exit(-1)
        if n_pos == 0 or n_unlab == 0 or n_features == 0 or n_classes < 2:
            traceback.print_stack()
            logging.error("wrong value passed for n_pos/n_unlab/n_features/n_classes. n_pos/n_unlab/n_features > 0 "
                          "and n_classes >1")
            exit(-1)
        if not scar and n_classes <= 2:
            traceback.print_stack()
            logging.error("number of classes should be >2 for SNAR data")
            exit(-1)

        # generate simulated data
        self.data, self.labels = make_classification(**self.params)
        # print the count for each class
        for c in np.unique(self.labels):
            logging.info("class {0} records: {1}".format(c, np.sum(self.labels == c)))

        # call sub-routines for SCAR and SNAR data
        if scar:
            logging.info("generate SCAR data with {0} positives and {1} unlabeled".format(n_pos, n_unlab))
            X, y_ml, y_true = self.generate_scar_pu_data(n_pos, n_unlab, pf_in_unlab, binarize=discrete_features)
        else:
            logging.info("generate SNAR data with {0} positives and {1} unlabeled".format(n_pos, n_unlab))
            X, y_ml, y_true = self.generate_non_scar_pu_data(n_pos, n_unlab, pf_in_unlab,
                                                             equal_frac_in_pos=equal_frac_in_pos,
                                                             binarize=discrete_features)
        return X, y_ml, y_true

    def generate_scar_pu_data(self, p_count, u_count, frac, binarize=False):
        """
        Generate PU dataset that satisfies the SCAR assumption.
        Keep one type of positive in both positive and unlabeled sets.
        1 - positive
        0 - unlabeled

        Parameters
        ----------
        p_count: number of positive records
        u_count: number of unlabeled records
        frac: fraction of positives in the unlabeled
        binarize: features binary or continuous?

        Returns
        ------
        X: simulated data containing positives and unlabeled
        y_ml: ML labels of the data
        y_orig: true labels of the data
        """

        # get indices of positive and unlabeled records (P=1 and U=0)
        random.seed(self.rseed)
        pos_0 = np.where(self.labels == 0)[0]
        random.shuffle(pos_0)
        pos_1 = np.where(self.labels == 1)[0]
        random.shuffle(pos_1)

        # record count in positive and unlabeled sets
        counts = [p_count, [round(u_count * frac), round((1 - frac) * u_count)]]

        # select indices for P and U records
        idx0 = np.concatenate([pos_1[:counts[1][0]], pos_0[:counts[1][1]]])  # unlabeled
        idx1 = pos_1[counts[1][0]:(counts[1][0] + counts[0])]  # positive
        # check if above logic worked fine
        if len(set(idx0).intersection(idx1)) > 0:
            traceback.print_stack()
            logging.error("Error in the data selection!!! PROGRAM WILL HALT")
            exit(-1)
        else:
            logging.info("P and U indices selected successfully")

        # combined indices and get original labels + data
        idx = np.concatenate([idx0, idx1])
        y_orig = self.labels[idx]
        for v in np.unique(y_orig):
            logging.info("PU data has {0} records from class {1}".format(np.sum(y_orig == v), v))
        y_ml = np.concatenate([[0] * len(idx0), [1] * len(idx1)])
        idx, y_ml, y_orig = shuffle(idx, y_ml, y_orig, random_state=self.rseed)

        # check if data need to be binarized
        if not binarize:
            return self.data[idx], y_ml, y_orig  # return data, PU label and true label
        else:
            return self.apply_binarization(self.data[idx]), y_ml, y_orig  # return data, PU label and true label

    def generate_non_scar_pu_data(self, p_count, u_count, frac, equal_frac_in_pos=True, binarize=False):
        """
        generate PU dataset that does not satisfy the SCAR assumption.
        keep multiple types of positives in both positive and unlabeled sets.
        1 - positive
        0 - unlabeled
        Assumption: data contains only one type of negative but many types of positive

        Parameters
        ----------
        p_count: number of positive records
        u_count: number of unlabeled records
        frac: fraction of positives in unlabeled
        equal_frac_in_pos: equal fraction of different types of positives in the positive set?
        binarize: feature value continuous or binary?

        Returns
        ------
        X: simulated data containing positives and unlabeled
        y_ml: ML labels of the data
        y_orig: true labels of the data
        """

        # generate a dict of "list of indices"
        idx_list = {}
        random.seed(self.rseed)
        unique_labels = sorted(np.unique(self.labels))
        self.p_types = len(np.unique(self.labels)) - 1
        for j in unique_labels:
            idx = np.where(self.labels == j)[0]
            random.shuffle(idx)
            idx_list[j] = idx
            # logging.info("class {0} has {1} records".format(j, len(idx)))

        # indices for positive set
        idx1 = []
        if equal_frac_in_pos:
            n_pos = [int(round(p_count / self.p_types))] * self.p_types
        else:
            n_pos = self.generate_k_randint(p_count)
        logging.info("positive counts in positive set: {0}".format(n_pos))
        # select indices
        i = 0
        for v in unique_labels[1:]:
            idx1.extend(idx_list[v][:n_pos[i]])
            i += 1

        # indices for unlabeled set
        control_count = int(u_count * (1 - frac))
        idx0 = list(idx_list[unique_labels[0]][:control_count])
        case_count = int(u_count * frac)
        # count of different types of positives in the unlabeled set
        p_count_in_unlab = self.different_pos_count_in_unlab(case_count)
        logging.info("positive counts in unlabeled set: {0}".format(p_count_in_unlab))
        # select indices
        i = 0
        for v in unique_labels[1:]:
            idx0.extend(idx_list[v][n_pos[i]: (n_pos[i] + p_count_in_unlab[i])])
            i += 1

        # check if above logic worked fine
        if len(set(idx0).intersection(idx1)) > 0:
            traceback.print_stack()
            logging.error("Error in the data selection!!! PROGRAM WILL HALT")
            exit(-1)
        else:
            logging.info("P and U indices selected successfully")

        # combine indices
        idx = np.concatenate([idx0, idx1])
        y_orig = self.labels[idx]
        for v in np.unique(y_orig):
            logging.info("PU data has {0} records from class {1}".format(np.sum(y_orig == v), v))
        y_ml = np.concatenate([[0] * len(idx0), [1] * len(idx1)])
        idx, y_ml, y_orig = shuffle(idx, y_ml, y_orig, random_state=self.rseed)

        # check if data need to be binarized
        if not binarize:
            return self.data[idx], y_ml, y_orig  # return data, PU label and true label
        else:
            return self.apply_binarization(self.data[idx]), y_ml, y_orig  # return data, PU label and true label

    def generate_k_randint(self, rec_count):
        """
        Use dirichlet or multinomial distribution to generate k random integers

        Parameters
        ----------
        rec_count: number of positive records

        Returns
        -------
        k_nums: number of different types of positives
        """

        # k_nums = np.random.dirichlet(np.ones(self.p_types), size=1)[0] * rec_count
        # k_nums = [round(k) for k in k_nums]
        k_nums = np.random.default_rng().multinomial(rec_count, [1 / self.p_types] * self.p_types, size=1)[0]
        logging.info("sum of random integers: {0}, {1}".format(np.sum(k_nums), rec_count))
        return k_nums

    def different_pos_count_in_unlab(self, case_count):
        """
        Compute the count of different types of positives in the unlabeled set
        fracs = [2^0/2^n-1, 2^1/2^n-1, 2^2/2^n-1, ..., 2^(n-1)/2^n-1], when n=positive types

        Parameters
        ----------
        case_count: number of positives in the unlabeled set

        Returns
        -------
        count of different types of positives in the unlabeled
        """

        # fraction of different types of positives in the unlabeled set
        p_frac_in_unlab = [2 ** i / (2 ** self.p_types - 1) for i in range(self.p_types)]
        return [round(case_count * v) for v in p_frac_in_unlab]

    def apply_binarization(self, xdata):
        """
        make_classification() generates continuous data. Binarize (0/1) the continuous data

        Parameters
        ----------
        xdata: feature matrix for ML

        Returns
        -------
        data with binary features
        """

        est = KBinsDiscretizer(n_bins=self.n_bins, encode=self.encode, strategy=self.strategy)
        est.fit(xdata)
        return est.transform(xdata)

    def generate_scar_pu_data_flip(self, p_count, u_count, frac, binarize=False):
        """
        Generate PU dataset that satisfies the SCAR assumption.
        Keep one type of positive in both positive an unlabeled set.
        0 - positive
        1 - unlabeled

        Parameters
        ----------
        p_count: number of positive records
        u_count: number of unlabeled records
        frac: fraction of positives in the unlabeled
        binarize: features binary or continuous?

        Returns
        ------
        X: simulated data containing positives and unlabeled
        y_ml: ML labels of the data
        y_orig: true labels of the data
        """

        print("Generate data with flipped labels")
        # get indices of positive and unlabeled records (P=1 and U=0)
        random.seed(self.rseed)
        pos_0 = np.where(self.labels == 0)[0]
        random.shuffle(pos_0)
        pos_1 = np.where(self.labels == 1)[0]
        random.shuffle(pos_1)
        logging.info("positive count: {0}, unlabeled count: {1}".format(len(pos_1), len(pos_0)))

        # record count in positive and unlabeled sets
        counts = [p_count, [round(u_count * frac), round((1 - frac) * u_count)]]

        # select indices for P and U records
        idx0 = np.concatenate([pos_0[:counts[1][1]], pos_1[:counts[1][0]]])  # unlabeled
        idx1 = pos_0[counts[1][1]:(counts[1][1] + counts[0])]  # positive
        # check if above logic worked fine
        if len(set(idx0).intersection(idx1)) > 0:
            traceback.print_stack()
            logging.error("Error in the data selection!!! PROGRAM WILL HALT")
            exit(-1)
        else:
            logging.info("P and U indices selected successfully")

        # combined indices
        idx = np.concatenate([idx0, idx1])
        y_orig = self.labels[idx]
        for v in np.unique(y_orig):
            logging.info("PU data has {0} records from class {1}".format(np.sum(y_orig == v), v))
        y_ml = np.concatenate([[1] * len(idx0), [0] * len(idx1)])
        idx, y_ml, y_orig = shuffle(idx, y_ml, y_orig, random_state=self.rseed)

        # check if data need to be binarized
        if not binarize:
            return self.data[idx], y_ml, y_orig  # return data, PU label and true label
        else:
            return self.apply_binarization(self.data[idx]), y_ml, y_orig  # return data, PU label and true label

    def generate_non_scar_pu_data_flip(self, p_count, u_count, frac, equal_frac_in_pos=True, binarize=False):
        """
        generate PU dataset that does not satisfy the SCAR assumption.
        keep multiple types of positives in both positive and unlabeled sets.
        0 - positive
        1 - unlabeled

        Parameters
        ----------
        p_count: number of positive records
        u_count: number of unlabeled records
        frac: fraction of positives in unlabeled
        equal_frac_in_pos: equal fraction of different types of positives in the positive set?
        binarize: feature value continuous or binary?

        Returns
        ------
        X: simulated data containing positives and unlabeled
        y_ml: ML labels of the data
        y_orig: true labels of the data
        """

        # generate a dict of "list of indices"
        idx_list = {}
        random.seed(self.rseed)
        unique_labels = sorted(np.unique(self.labels))
        self.p_types = len(np.unique(self.labels)) - 1
        for j in unique_labels:
            idx = np.where(self.labels == j)[0]
            random.shuffle(idx)
            idx_list[j] = idx
            logging.info("class {0} has {1} records".format(j, len(idx)))

        # indices for unlabeled set
        control_count = int(u_count * (1 - frac))
        idx0 = list(idx_list[unique_labels[0]][:control_count])
        case_count = int(u_count * frac)
        # count of different types of positives in the unlabeled set
        p_count_in_unlab = self.different_pos_count_in_unlab(case_count)
        logging.info("positive counts in unlabeled set: {0}".format(p_count_in_unlab))
        # select indices
        i = 0
        for v in unique_labels[1:]:
            idx0.extend(idx_list[v][:p_count_in_unlab[i]])
            i += 1

        # indices for positive set
        idx1 = list(idx_list[unique_labels[0]][control_count: (control_count + p_count)])
        logging.info("idx0 and idx1 length: {0}, {1}".format(len(idx0), len(idx1)))
        # check if above logic worked fine
        if len(set(idx0).intersection(idx1)) > 0:
            traceback.print_stack()
            logging.error("Error in the data selection!!! PROGRAM WILL HALT")
            exit(-1)
        else:
            logging.info("P and U indices selected successfully")

        # combine indices
        idx = np.concatenate([idx0, idx1])
        y_orig = self.labels[idx]
        for v in np.unique(y_orig):
            logging.info("PU data has {0} records from class {1}".format(np.sum(y_orig == v), v))
        y_ml = np.concatenate([[1] * len(idx0), [0] * len(idx1)])
        idx, y_ml, y_orig = shuffle(idx, y_ml, y_orig)

        # check if data need to be binarized
        if not binarize:
            return self.data[idx], y_ml, y_orig  # return data, PU label and true label
        else:
            return self.apply_binarization(self.data[idx]), y_ml, y_orig  # return data, PU label and true label
