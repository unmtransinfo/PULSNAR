import pickle
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve  # , CalibrationDisplay
import logging, traceback

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


class DataFileUtils:
    def __init__(self, datafile, ml_labelfile, tru_labelfile):
        self.datafile = datafile
        self.ml_labelfile = ml_labelfile
        self.tru_labelfile = tru_labelfile

    def save_data_label(self, data, ml_label, tru_label, p_protocol=4):
        """
        Save data and label for future testing

        Parameters
        ----------
        data: feature matrix for ML
        ml_label: ML labels
        true_label: true labels (only for known datasets such simulated data)
        p_protocol: pickle protocol. select it as per your system. Default is 5 for Python 3.8
        """

        # data
        with open(self.datafile, "wb") as fh1:
            pickle.dump(data, fh1, protocol=p_protocol)

        # ML label
        with open(self.ml_labelfile, "wb") as fh2:
            pickle.dump(ml_label, fh2, protocol=p_protocol)

        # true label
        with open(self.tru_labelfile, "wb") as fh3:
            pickle.dump(tru_label, fh3, protocol=p_protocol)

    def read_data_label(self):
        """
        Read data and label from saved pickle files.

        Returns
        -------
        X: feature matrix for ML
        y_ml: ML labels
        y_true: true labels
        """
        # data
        with open(self.datafile, "rb") as fh1:
            X = pickle.load(fh1)

        # ML label
        with open(self.ml_labelfile, "rb") as fh2:
            y_ml = pickle.load(fh2)

        # true label
        with open(self.tru_labelfile, "rb") as fh3:
            y_true = pickle.load(fh3)

        return X, y_ml, y_true


class ClassificationFileUtils:
    def __init__(self, ofile=None, scar=True, keepalpha=True, alphafile=None):
        """
        Write header to the output file depending on the type of data
        """
        self.ofile = ofile
        self.scar = scar
        self.keepalpha = keepalpha
        self.alphafile = alphafile
        if self.ofile is None:
            traceback.print_stack()
            logging.error("output file for classification results missing")
            exit(-1)

        # open classification results file
        self.fobj = open(ofile, 'w')
        if scar:
            hdr = "rec_id" + "\t" + "true_label" + "\t" + "ml_label" + "\t" + "predicted_prob" + "\t" + \
                  "calibrated_prob" + "\t" + "positive_count" + "\t" + "unlabeled_count" + "\t" + "iteration" + "\t" + \
                  "pos_frac_in_unlabeled" + "\n"
        else:
            hdr = "rec_id" + "\t" + "true_label" + "\t" + "ml_label" + "\t" + "predicted_prob" + "\t" + \
                  "calibrated_prob" + "\t" + "positive_count" + "\t" + "unlabeled_count" + "\t" + "iteration" + "\t" + \
                  "cluster" + "\t" + "pos_frac_in_unlabeled" + "\n"
        self.fobj.write(hdr)

        # create a file to store alpha estimates if it is required
        if self.keepalpha:
            if self.alphafile is None:
                traceback.print_stack()
                logging.error("output file for storing estimates missing")
                exit(-1)
            else:
                alpha_hdr = "cluster" + "\t" + "iteration" + "\t" + "true alpha" + "\t" + "estimated alpha" + "\n"
                self.fobjalpha = open(self.alphafile, 'w')
                self.fobjalpha.write(alpha_hdr)

    def write_ml_prediction_results(self, p_count, u_count, ml_label, preds, tru_label, rec_ids, iteration,
                                    calibrated_preds, cluster='', p_frac=''):
        """
        This function writes classification results for the given data
        """
        if self.scar:
            for j in range(len(tru_label)):
                line = str(rec_ids[j]) + "\t" + str(tru_label[j]) + "\t" + str(ml_label[j]) + "\t" + \
                       str(preds[j]) + "\t" + str(calibrated_preds[j]) + "\t" + str(p_count) + "\t" + \
                       str(u_count) + "\t" + str(iteration) + "\t" + str(p_frac) + "\n"
                self.fobj.write(line)
            self.fobj.flush()
        else:
            for j in range(len(tru_label)):
                line = str(rec_ids[j]) + "\t" + str(tru_label[j]) + "\t" + str(ml_label[j]) + "\t" + \
                       str(preds[j]) + "\t" + str(calibrated_preds[j]) + "\t" + str(p_count) + "\t" + \
                       str(u_count) + "\t" + str(iteration) + "\t" + str(cluster) + "\t" + str(p_frac) + "\n"
                self.fobj.write(line)
            self.fobj.flush()

    def write_alpha_estimates(self, true_alpha, est_alpha, rep, cluster=''):
        """
        This function writes alpha estimates to a file
        """
        line = str(cluster) + "\t" + str(rep) + "\t" + str(true_alpha) + "\t" + str(est_alpha) + "\n"
        self.fobjalpha.write(line)

    def close_open_files(self):
        if self.ofile is not None:
            self.fobj.close()
        if self.keepalpha:
            self.fobjalpha.close()


class CalibrationUtils:
    def __init__(self, ifile=None, ofile=None, pfile=None, sep="\t", hdr_pos=0, n_bins=10, pformat="png", pdpi=300):
        if pfile is None or ifile is None:
            traceback.print_stack()
            logging.error("input, and plot files are mandatory")
            exit(-1)
        else:
            self.ifile = ifile
            self.ofile = ofile
            self.pfile = pfile
        self.sep = sep
        self.hdr_pos = hdr_pos
        self.n_bins = n_bins
        self.pformat = pformat
        self.pdpi = pdpi

    def get_classification_data(self):
        """
        This function reads the classification results file and returns a dataframe. It also saves the average
        classification results.

        Returns
        -------
        a dataframe
        """

        # read input data as a dataframe
        dat = pd.read_csv(self.ifile, sep=self.sep, header=self.hdr_pos)
        # dat = dat.drop(columns=['flipped_ml_label', 'positive_count', 'unlabeled_count', 'iteration',
        #                        'pos_frac_in_unlabeled'])
        # dat = dat.groupby(['rec_id']).mean()
        # dat.reset_index(inplace=True)

        # save the mean results
        # dat.to_csv(self.ofile, index=False, sep="\t")
        return dat

    def generate_calibration_curve(self):
        """
        This function generates the calibration curve using the average probabilities over k iterations.
        """

        logging.info("read input data and generate calibration curve")
        rdata = self.get_classification_data()
        iterations = np.unique(rdata['iteration'].to_numpy())
        fig, ax = plt.subplots()

        for itr in iterations:
            dat = rdata[rdata['iteration'] == itr]
            # generate calibration curve
            dat['true_label'][dat['true_label'] >= 1] = 1  # set labels of all positives to 1
            prob_true, prob_pred = calibration_curve(dat['true_label'].to_numpy(), dat['calibrated_prob'].to_numpy(),
                                                     n_bins=self.n_bins)
            plt.plot(prob_pred, prob_true, marker='o')
            # CalibrationDisplay.from_predictions(dat['true_label'].to_numpy(), dat['calibrated_prob'].to_numpy(),
            #                                    n_bins=self.n_bins)
        plt.plot([0, 1], [0, 1])
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration curve")
        fig.tight_layout()
        plt.savefig(self.pfile, dpi=self.pdpi, format=self.pformat)
        # plt.show()


class MiscUtils:
    def __init__(self, bic_plot=False, ci_mean=False):
        self.bic_plot = bic_plot
        self.ci_mean = ci_mean

    def draw_line_plot(self, xdata, ydata, pfile, val='BIC'):
        """
        Draw BIC/AIC vs cluster count plot.

        Parameters
        ---------
        xdata: cluster count
        ydata: BIC/AIC
        pfile: file to save the plot
        val: BIC/AIC values
        """
        if self.bic_plot:
            plt.plot(xdata, ydata, 'g-o')
            plt.ylabel(val)
            plt.xlabel("Cluster count")
            plt.savefig(pfile, dpi=300)

    def compute_mean_and_confidence_interval(self, data, ci=0.95):
        """
        compute mean and confidence interval using the given data

        Parameters
        ----------
        data: a list of values
        ci: confidence interval

        Returns
        -------
        avg: mean of values
        lv: lower value of confidence interval
        uv: upper value of confidence interval
        """
        if self.ci_mean:
            avg = np.mean(data)
            lv, uv = st.t.interval(alpha=ci, df=len(data) - 1, loc=avg, scale=st.sem(data))
            return avg, lv, uv
