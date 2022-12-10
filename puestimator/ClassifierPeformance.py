import logging
import os
import traceback
import numpy as np
import yaml
from PULSNAR.puestimator import PulsnarParams as pp
from PULSNAR.puestimator.MLEstimators import ClassificationEstimator
from PULSNAR.puestimator.ProbabilityCalibration import CalibrateProbabilities
from PULSNAR.puutils.DataPreProcessor import MLDataPreprocessing
from scipy import sparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle


def pulsnar_performance_metrics(preds, y_orig, scar=True):
    """
    This function computes several classification metrics. The predictions are based on labeled positives, probable
    positives and the rest of the unlabeled records.
    """
    if not scar:
        y_orig[y_orig > 1] = 1

    # classification metrics
    after_bs = brier_score_loss(y_orig, preds)
    after_aps = average_precision_score(y_orig, preds)
    after_auc = roc_auc_score(y_orig, preds)
    after_f1 = f1_score(y_orig, np.round(preds).astype(int))
    after_mcc = matthews_corrcoef(y_orig, np.round(preds).astype(int))
    after_acc = accuracy_score(y_orig, np.round(preds).astype(int))

    return after_bs, after_aps, after_auc, after_f1, after_mcc, after_acc


# *** Methods to return calibrated probabilities ****
class MLPerformanceEvaluation:
    def __init__(self, data=None, label=None, tru_label=None, all_rec_ids=None, calibration_method='isotonic',
                 csrdata=False, k_flips=20, n_bins=100, alpha=None, classifier='xgboost', clf_params_file=None):

        # set variables
        if data is None:
            traceback.print_stack()
            logging.error("please provide the feature matrix")
            exit(-1)
        else:
            self.data = data

        if label is None:
            traceback.print_stack()
            logging.error("please provide the labels of the data")
            exit(-1)
        else:
            self.label = label

        if tru_label is None:
            traceback.print_stack()
            logging.error("please provide the true labels of the data")
            exit(-1)
        else:
            self.tru_label = tru_label

        if all_rec_ids is None:
            traceback.print_stack()
            logging.error("please provide record ids")
            exit(-1)
        else:
            self.all_rec_ids = all_rec_ids

        self.n_bins = n_bins

        if alpha is None:
            traceback.print_stack()
            logging.error("please provide the estimated fraction (alpha)")
            exit(-1)
        else:
            self.alpha = alpha

        self.calibration_method = calibration_method
        self.csrdata = csrdata
        self.k_flips = k_flips
        self.classifier = classifier
        self.clf_params_file = clf_params_file

    def prediction_using_probable_positives(self, preds, y_ml, y_orig, rec_ids):
        """
        Run the ML using probable positives, labeled positives and the remaining of the unlabeled records 
        """
        # calibrated predicted probabilities of only unlabeled records
        cal_probs = CalibrateProbabilities(calibration_data='U', n_bins=self.n_bins, alpha=self.alpha,
                                           calibration_method=self.calibration_method, smooth_isotonic=False)

        u_preds, u_y_ml, u_y_orig, u_rec_ids, u_calibrated_preds = \
            cal_probs.calibrate_predicted_probabilities(preds, y_ml, y_orig, rec_ids, k_flips=self.k_flips)

        # find probable positives among unlabeled and update ml labels
        X, Y, Y_true, rec_list = self.identify_probable_positives(u_rec_ids, u_calibrated_preds)

        # run ML to get predictions
        X, Y, Y_true, rec_list = shuffle(X, Y, Y_true, rec_list, random_state=123)
        clf_params = self.get_params()
        ml_clf = ClassificationEstimator(clf=self.classifier, clf_params=clf_params)
        preds, y_ml, y_orig, recs, _ = ml_clf.train_test_model(X, Y, Y_true, rec_list, k_folds=5,
                                                               rseed=123, calibration=None)
        return preds, y_orig, recs

    def identify_probable_positives(self, u_rec_ids, u_calibrated_preds):
        """
        using calibrated probabilities, this function determines the probable positives (records with higher calibrated
        probabilities) among unlabeled examples.
        """
        # split data into positive and unlabeled sets
        ml_data = MLDataPreprocessing(rseed=123)
        X_pos, _, y_true_pos, rec_pos, X_unlab, _, y_true_unlab, rec_unlab = \
            ml_data.generate_pu_dataset(self.data, self.label, self.tru_label, self.all_rec_ids)

        # flip labels of top alpha*U elements from 0 to 1
        ix = np.argsort(u_calibrated_preds)[::-1]  # indices sorted in descending order of cal_probs
        u_rec_ids = u_rec_ids[ix]  # sort rec_ids by decreasing calibrated probabilities

        # rearrange data according to the shuffled record ids
        u_rec_list_indx = dict(zip(rec_unlab, range(len(rec_unlab))))
        indx = [u_rec_list_indx[r] for r in u_rec_ids]
        X_unlab = X_unlab[indx]
        y_true_unlab = y_true_unlab[indx]
        rec_unlab = rec_unlab[indx]

        # rearrange data for ML
        if self.csrdata:
            X = sparse.vstack((X_pos, X_unlab), format='csr')
        else:
            X = np.concatenate([X_pos, X_unlab])
        rec_list = np.concatenate([rec_pos, rec_unlab])
        Y = np.concatenate(
            [[1] * X_pos.shape[0], [1] * int(self.alpha * X_unlab.shape[0]),
             [0] * (X_unlab.shape[0] - int(self.alpha * X_unlab.shape[0]))])
        Y_true = np.concatenate([y_true_pos, y_true_unlab])

        return X, Y, Y_true, rec_list

    def get_params(self):
        """
        Check if user provided parameters for the classifier. If not, get the default value from the
        pulsnar_args.ini file
        """
        # check if param file was provided by user?
        param_file = False
        config = None
        if self.clf_params_file is not None and os.path.exists(self.clf_params_file):
            # read the parameters file
            with open(self.clf_params_file, 'r') as fyml:
                config = yaml.safe_load(fyml)
                param_file = True

        # classifier parameters
        if self.classifier == 'xgboost':
            if param_file:
                return dict(config['XGB_params'])
            else:
                return pp.XGB_params
        elif self.classifier == 'catboost':
            if param_file:
                return dict(config['CB_params'])
            else:
                return pp.CB_params
        elif self.classifier == 'LR':
            if param_file:
                return dict(config['LR_params'])
            else:
                return pp.LR_params
        else:
            traceback.print_stack()
            logging.error("Unknown classifier!!!, currently PULSNAR supports XGBoost, CatBoost "
                          "and Logistic Regression. supported values are: ['xgboost', 'catboost', 'LR']")
            exit(-1)
