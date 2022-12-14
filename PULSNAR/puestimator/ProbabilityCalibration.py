import logging
import traceback
from copy import deepcopy
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


# *** Methods to return calibrated probabilities ****
class CalibrateProbabilities:
    def __init__(self, calibration_data='PU', n_bins=100, alpha=None, calibration_method='sigmoid',
                 smooth_isotonic=False):
        self.calibration_data = calibration_data
        self.n_bins = n_bins
        self.calibration_method = calibration_method
        self.smooth_isotonic = smooth_isotonic

        if alpha is None:
            traceback.print_stack()
            logging.error("please provide the estimated fraction (alpha)")
            exit(-1)
        else:
            self.alpha = alpha

    def calibrate_predicted_probabilities(self, predicted_probs, ml_label, true_label, shuffled_recs, k_flips=1):
        """
        This functions flips the labels of some of the unlabeled records and then apply IsotonicRegression or
        LogisiticRegression to get the calibrated predicted probabilities.

        Parameters
        ----------
        predicted_probs: list of ML predicted probabilities
        ml_label: machine learning labels of records
        true_label: true labels of the records (only applicable for known data such as simulated data)
        shuffled_recs: shuffled record ids
        k_flips: number of times label flipping should be done

        Returns
        -------
        probs: list of ML predicted probabilities
        Y_true: true labels of the records (only applicable for known data such as simulated data)
        rec_list: list of record ids
        norm_s_probs: calibrated probabilities
        Ys: original ML labels
        """

        # divide records into positive and unlabeled groups
        # positive group
        idx1 = np.where(ml_label == 1)[0]
        orig_y_ml1 = ml_label[idx1]
        orig_y_true1 = true_label[idx1]
        orig_probs1 = predicted_probs[idx1]
        orig_recs1 = shuffled_recs[idx1]

        # unlabeled group
        idx0 = np.where(ml_label == 0)[0]
        orig_y_ml0 = ml_label[idx0]
        orig_y_true0 = true_label[idx0]
        orig_probs0 = predicted_probs[idx0]
        orig_recs0 = shuffled_recs[idx0]

        # flip labels of some of unlabeled records
        calibrated_probs = 0
        for k_flip in range(k_flips):
            y_ml1, y_true1, probs1, recs1 = deepcopy(orig_y_ml1), deepcopy(orig_y_true1), deepcopy(
                orig_probs1), deepcopy(orig_recs1)
            y_ml0, y_true0, probs0, recs0 = deepcopy(orig_y_ml0), deepcopy(orig_y_true0), deepcopy(
                orig_probs0), deepcopy(orig_recs0)

            # flip labels
            y_ml0 = self.flip_labels_of_unlabs(probs1, probs0, y_ml0, k_flip)

            # generate data for isotonic calibration
            probs, Y, Ys, Y_true, rec_list = None, None, None, None, None
            if self.calibration_data == 'PU':
                probs = np.concatenate([probs1, probs0])
                Y = np.concatenate([y_ml1, y_ml0])  # flipped ML labels
                Ys = np.concatenate([y_ml1, [0] * len(y_ml0)])  # original ML labels
                Y_true = np.concatenate([y_true1, y_true0])
                rec_list = np.concatenate([recs1, recs0])
            elif self.calibration_data == 'U':
                probs = probs0
                Y = y_ml0  # flipped ML labels
                Ys = np.asarray([0] * len(y_ml0))  # original ML labels
                Y_true = y_true0
                rec_list = recs0

            # apply isotonic or logistic regression for calibration
            if self.calibration_method == 'isotonic':
                ix = np.argsort(probs)  # sort predicted probability
                # apply isotonic regression on probs and updated labels
                reg = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip').fit(probs[ix], Y[ix])
                iso_probs = reg.predict(probs)
                if self.smooth_isotonic:
                    # smooth isotonic calibrated probabilities
                    spl = UnivariateSpline(probs[ix], iso_probs[ix], k=3)
                    s_probs = spl(probs)
                    minsp = min(s_probs)
                    maxsp = max(s_probs)
                    norm_s_probs = np.asarray([(v - minsp) / (maxsp - minsp) for v in s_probs])
                    calibrated_probs += norm_s_probs
                else:
                    calibrated_probs += iso_probs
            else:
                ix = np.argsort(probs)  # sort predicted probability
                # apply logistic regression on probs and updated labels
                reg = LogisticRegression().fit(probs[ix].reshape(-1, 1), Y[ix])
                calibrated_probs += reg.predict_proba(probs.reshape(-1, 1))[:, 1]

        # prepare returned values
        if self.smooth_isotonic:
            if sum(calibrated_probs / k_flips) > sum(iso_probs):
                calibrated_probs = (calibrated_probs / k_flips) * sum(iso_probs) / sum(calibrated_probs / k_flips)
            else:
                calibrated_probs = calibrated_probs / k_flips
        else:
            calibrated_probs = calibrated_probs / k_flips
        # print("sum of probs (original, calibrated): ", np.sum(probs), np.sum(calibrated_probs))
        return probs, Ys, Y_true, rec_list, calibrated_probs

    def flip_labels_of_unlabs(self, pos_probs, unlab_probs, y, rs):
        """
        This function flips the labels of some of the unlabeled records. flipping is done using the density of the
        probabilities of positive records.

        Parameters
        -----------
        pos_probs: list of probabilities of positive records
        unlab_probs: list of probabilities of unlabeled records
        y: ML labels of the unlabeled records

        Returns
        -------
        y: flipped ML labels
        """
        # determine how many labels need to be flipped in each bin
        np.random.seed(rs)
        bin_edges = [ii / self.n_bins for ii in range(self.n_bins + 1)]
        bin_count = self.how_many_labels_to_flip_in_each_bin(pos_probs, unlab_probs, bin_edges)
        # logging.info("sum of ml labels of unlabeled before flipping: {0}".format(sum(y)))

        # start flipping labels
        diff = 0
        for j in range(len(bin_edges) - 1, 0, -1):
            flip_count = bin_count[j - 1] + diff
            indx = np.where((unlab_probs >= bin_edges[j - 1]) & (unlab_probs < bin_edges[j]))[0]

            # check if bins do not have enough records
            if len(indx) < flip_count:
                kk = len(indx)
                diff = flip_count - len(indx)
            else:
                kk = flip_count
                diff = 0
            # make sure bin has records
            if kk > 0:
                i = np.random.choice(indx, kk, replace=False)
                # i = indx[-1 * kk:]  # change from right -> left
                y[i] = 1
        # logging.info("sum of ml labels of unlabeled after flipping: {0}".format(sum(y)))
        return y

    def how_many_labels_to_flip_in_each_bin(self, pos, unlab, s):
        """
        Using histogram (density) for positive records, determine how many labels need to be flipped in each bin of
        the unlabeled records.

        Parameters
        ----------
        pos: list of probabilities of positive records
        unlab: list of probabilities of unlabeled records
        s: bins

        Returns
        -------
        u_bin_count: number of labels that need to be flipped in each bin
        """
        p_bin_count, _ = np.histogram(pos, bins=s)
        proportion = p_bin_count / np.sum(p_bin_count)  # proportion for each bin

        # find bin_counts for unlabeled
        p_count_in_unlab = int(len(unlab) * self.alpha)
        u_bin_count = p_count_in_unlab * proportion
        u_bin_count = u_bin_count.astype(int)  # convert to int

        # check if bin_count = p_count in unlab
        if sum(u_bin_count) < p_count_in_unlab:
            diff = p_count_in_unlab - sum(u_bin_count)
            k = len(u_bin_count) - 1
            while diff > 0:
                u_bin_count[k] += 1
                diff -= 1
                k -= 1
                if k == 0:
                    k = len(u_bin_count) - 1

        # logging.info("number of records to be flipped in each bin: {0}".format(u_bin_count))
        # logging.info("total number records to be flipped: {0}, number of bins: {1}".format(sum(u_bin_count),
        #                                                                                   len(u_bin_count)))
        return u_bin_count
