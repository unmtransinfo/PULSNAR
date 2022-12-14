import os
import logging
import traceback
import numpy as np
import yaml
from sklearn.utils import shuffle
from collections import Counter
import pickle
from scipy import sparse
from copy import deepcopy

from PULSNAR.puestimator.MLEstimators import ClassificationEstimator, ClusteringEstimator
from PULSNAR.puestimator.ProbabilityCalibration import CalibrateProbabilities
from PULSNAR.puestimator.AlphaEstimate import PositiveFractionEstimation
from PULSNAR.puestimator.PosteriorEstimates import PosteriorEstimation
from PULSNAR.puestimator.ClassifierPeformance import MLPerformanceEvaluation, pulsnar_performance_metrics
from PULSNAR.puutils.AllUtils import ClassificationFileUtils, MiscUtils, CalibrationUtils
from PULSNAR.puutils.DataPreProcessor import MLDataPreprocessing
from PULSNAR.puestimator import PulsnarParams as pp


class PULSNARClassifier:
    def __init__(self, scar=True, csrdata=False, classifier='xgboost', n_clusters=0, max_clusters=25, covar_type='full',
                 top50p_covars=False, bin_method='scott', bw_method='hist', lowerbw=0.01, upperbw=0.5, optim='global',
                 calibration=False, calibration_data='PU', calibration_method='isotonic', calibration_n_bins=100,
                 smooth_isotonic=False, classification_metrics=False, n_iterations=1, kfold=5, kflips=1,
                 pulsnar_params_file=None):

        # set class variables
        self.scar = scar
        self.csrdata = csrdata
        self.classifier = classifier
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.covar_type = covar_type
        self.top50p_covars = top50p_covars
        self.bin_method = bin_method
        self.bw_method = bw_method
        self.lowerbw = lowerbw
        self.upperbw = upperbw
        self.optim = optim
        self.calibration = calibration
        self.calibration_data = calibration_data
        self.calibration_method = calibration_method
        self.calibration_n_bins = calibration_n_bins
        self.smooth_isotonic = smooth_isotonic
        self.classification_metrics = classification_metrics
        self.n_iterations = n_iterations
        self.kfold = kfold
        self.kflips = kflips
        self.pulsnar_params_file = pulsnar_params_file

    def pulsnar(self, data, label, tru_label=None, rec_list=None):
        """
        This is the main driver function. It uses the arguments passed to determine the sub-routines to invoke
        and returns results: estimated alpha, file containing predicted probabilities, file containing alpha values,
        file containing BIC vs cluster count plot, file containing ML model's important features, and classification
        performance metrics.

        parameters
        ----------
        data:  feature matrix (X values)
        label: class labels of the data
        tru_label: true class labels of the data (only applicable for the known datasets)
        rec_list: list of record ids
        """
        # check if user provided true labels or not
        if tru_label is None:
            if self.classification_metrics:
                logging.error("classification performance metrics are calculated only if true labels are known. "
                              "since true labels are not given, performance metrics will not be calculated.")
                self.classification_metrics = False
            tru_label = label

        # process SCAR or SNAR data
        logging.info("PULSNAR will return estimated alpha, output files, and classification performance metrics "
                     "if true labels are provided")
        if self.scar:
            return self.scar_data_processing(data, label, tru_label, rec_list)
        else:
            return self.snar_data_processing(data, label, tru_label, rec_list)

    def scar_data_processing(self, X, Y, Y_true, mv_list):
        """
        Process scar data and return results
        """
        # local variables to store results
        est_alphas = []
        res = {}
        clf_bs = []
        clf_aps = []
        clf_auc = []
        clf_f1 = []
        clf_mcc = []
        clf_acc = []

        # instantiate file operation class
        io_files = self.get_params(option='IO')
        r_fileop = ClassificationFileUtils(ofile=io_files['result_file'], scar=True, keepalpha=True,
                                           alphafile=io_files['alpha_file'])
        # run PULSCAR and get results
        for itr in range(self.n_iterations):
            logging.info("Estimating alpha for iteration {0}".format(itr+1))
            X, Y, Y_true, mv_list = shuffle(X, Y, Y_true, mv_list, random_state=itr)
            preds, y_ml, y_orig, rec_ids, est_alpha = self.run_ml_and_estimate_alpha(X, Y, Y_true, mv_list, itr)
            # store the estimated alpha
            est_alphas.append(est_alpha)

            # probability calibration if needed
            if self.calibration:
                logging.info("Calibrating predicted probabilities for iteration {0}".format(itr + 1))
                preds, y_ml, y_orig, rec_ids, calibrated_preds = self.apply_calibration(preds, y_ml, y_orig, rec_ids,
                                                                                        est_alpha)
            else:
                calibrated_preds = preds

            # write classification predictions to an output file
            rec_count = Counter(y_ml)
            r_fileop.write_ml_prediction_results(rec_count[1], rec_count[0], y_ml, preds, y_orig, rec_ids, itr,
                                                 calibrated_preds, cluster='', p_frac='')

            # write alpha estimates to an output file
            r_fileop.write_alpha_estimates('', est_alpha, itr, cluster='')

            # calculate classification performance metrics
            if self.classification_metrics:
                logging.info("Estimating classification performance metrics for iteration {0}".format(itr + 1))
                mlpe = MLPerformanceEvaluation(data=X, label=Y, tru_label=Y_true, all_rec_ids=mv_list,
                                               calibration_method=self.calibration_method, csrdata=self.csrdata,
                                               k_flips=self.kflips, n_bins=self.calibration_n_bins, alpha=est_alpha,
                                               classifier=self.classifier, clf_params_file=self.pulsnar_params_file)

                cls_metrics_preds, cls_metrics_y_true, _ = mlpe.prediction_using_probable_positives(preds, y_ml, y_orig,
                                                                                                    rec_ids)
                itr_bs, itr_aps, itr_auc, itr_f1, itr_mcc, itr_acc = \
                    pulsnar_performance_metrics(cls_metrics_preds, cls_metrics_y_true, scar=True)

                # store performance metrics
                clf_bs.append(itr_bs)
                clf_aps.append(itr_aps)
                clf_auc.append(itr_auc)
                clf_f1.append(itr_f1)
                clf_mcc.append(itr_mcc)
                clf_acc.append(itr_acc)

        # return estimated alpha and output files
        res['estimated_alpha'] = np.mean(est_alphas)
        res['prediction_file'] = io_files['result_file']
        res['alpha_file'] = io_files['alpha_file']
        if self.classification_metrics:
            res['pulsnar_brier_score'] = np.mean(clf_bs)
            res['pulsnar_average_precision_score'] = np.mean(clf_aps)
            res['pulsnar_auc_roc_score'] = np.mean(clf_auc)
            res['pulsnar_f1_score'] = np.mean(clf_f1)
            res['pulsnar_mcc_score'] = np.mean(clf_mcc)
            res['pulsnar_accuracy'] = np.mean(clf_acc)

        return res

    def snar_data_processing(self, X, Y, Y_true, mv_list):
        """
        Process scar data and return results
        """
        # local variables to store results
        est_alphas = []
        res = {}
        clf_bs = []
        clf_aps = []
        clf_auc = []
        clf_f1 = []
        clf_mcc = []
        clf_acc = []

        # instantiate file operation class
        io_files = self.get_params(option='IO')
        r_fileop = ClassificationFileUtils(ofile=io_files['result_file'], scar=False, keepalpha=True,
                                           alphafile=io_files['alpha_file'])

        # get important features using all data
        logging.info("Finding important features for the ML model")
        X, Y, Y_true, mv_list = shuffle(X, Y, Y_true, mv_list, random_state=123)
        impf = self.get_model_imp_features(X, Y)

        '''
        if os.path.exists(pulsnar_params['imp_feature_file']):
            with open(pulsnar_params['imp_feature_file'], 'rb') as fh:
                impf = pickle.load(fh)
        else:
            impf = get_model_imp_features(X, Y)
            # save important features
            with open(pulsnar_params['imp_feature_file'], 'wb') as fh:
                pickle.dump(impf, fh, protocol=4)
        '''
        # divide data into positive and unlabeled sets
        ml_data = MLDataPreprocessing(rseed=123)
        X_pos, y_ml_pos, y_true_pos, mv_pos, X_unlab, y_ml_unlab, y_true_unlab, mv_unlab = \
            ml_data.generate_pu_dataset(X, Y, Y_true, mv_list)

        # divide positive data into clusters
        logging.info("Diving positives into k clusters")
        if self.n_clusters == 0:
            clster_indx, f_idx = self.determine_clusters(impf, X_pos, self.covar_type, io_files['bic_plot_file'],
                                                         n_clusters=None, csr=self.csrdata, top50p=self.top50p_covars)
        else:
            clster_indx, f_idx = self.determine_clusters(impf, X_pos, self.covar_type, io_files['bic_plot_file'],
                                                         n_clusters=self.n_clusters, csr=self.csrdata,
                                                         top50p=self.top50p_covars)
        # use only important features.
        X_pos, X_unlab = X_pos[:, f_idx], X_unlab[:, f_idx]

        # run ML model per cluster
        for itr in range(self.n_iterations):
            sl = 0  # cluster number
            alpha_list = []
            rec_preds = {}
            rec_cal_preds = {}
            rec_y_ml = {}
            rec_y_orig = {}
            cls_metrics_rec_preds = {}
            cls_metrics_rec_y_true = {}

            # go through each cluster
            for idx1 in clster_indx:
                sl += 1
                if self.csrdata:
                    X_cluster = sparse.vstack((X_pos[idx1], X_unlab), format='csr')
                else:
                    X_cluster = np.vstack([X_pos[idx1], X_unlab])
                y_ml_cluster = np.concatenate([y_ml_pos[idx1], y_ml_unlab])
                mv_cluster = np.concatenate([mv_pos[idx1], mv_unlab])
                y_true_cluster = np.concatenate([y_true_pos[idx1], y_true_unlab])

                # shuffle the data
                X_cluster, y_ml_cluster, y_true_cluster, mv_cluster = shuffle(X_cluster, y_ml_cluster, y_true_cluster,
                                                                              mv_cluster, random_state=itr)

                logging.info("Estimating alpha for iteration {0} and cluster {1}".format(itr + 1, sl))
                preds, y_ml, y_orig, rec_ids, est_alpha = self.run_ml_and_estimate_alpha(X_cluster, y_ml_cluster,
                                                                                         y_true_cluster,
                                                                                         mv_cluster, itr)
                # store the estimated alpha
                alpha_list.append(est_alpha)
                # write alpha estimates to an output file
                r_fileop.write_alpha_estimates('', est_alpha, itr, cluster=sl)

                # probability calibration if needed
                if self.calibration:
                    logging.info("Calibrating predicted probabilities for iteration {0} and cluster {1}".
                                 format(itr + 1, sl))
                    preds, y_ml, y_orig, rec_ids, calibrated_preds = self.apply_calibration(preds, y_ml, y_orig,
                                                                                            rec_ids, est_alpha)
                else:
                    calibrated_preds = preds

                # store predictions in a dictionary to compute the posterior using k cluster
                for i, r in enumerate(rec_ids):
                    if r in rec_preds:
                        rec_preds[r].append(preds[i])
                        rec_y_ml[r].append(y_ml[i])
                        rec_cal_preds[r].append(calibrated_preds[i])
                        rec_y_orig[r].append(y_orig[i])
                    else:
                        rec_preds[r] = [preds[i]]
                        rec_cal_preds[r] = [calibrated_preds[i]]
                        rec_y_ml[r] = [y_ml[i]]
                        rec_y_orig[r] = [y_orig[i]]

                # calculate classification performance metrics
                if self.classification_metrics:
                    logging.info("Estimating classification performance metrics for iteration {0} and cluster {1}".
                                 format(itr + 1, sl))
                    mlpe = MLPerformanceEvaluation(data=X_cluster, label=y_ml_cluster, tru_label=y_true_cluster,
                                                   all_rec_ids=mv_cluster, calibration_method=self.calibration_method,
                                                   csrdata=self.csrdata, k_flips=self.kflips,
                                                   n_bins=self.calibration_n_bins, alpha=est_alpha,
                                                   classifier=self.classifier, clf_params_file=self.pulsnar_params_file)

                    cls_metrics_preds, cls_metrics_y_true, cls_metrics_recs = \
                        mlpe.prediction_using_probable_positives(preds, y_ml, y_orig, rec_ids)

                    # store prediction to compute posterior
                    for i, r in enumerate(cls_metrics_recs):
                        if r in cls_metrics_rec_preds:
                            cls_metrics_rec_preds[r].append(cls_metrics_preds[i])
                            cls_metrics_rec_y_true[r].append(cls_metrics_y_true[i])
                        else:
                            cls_metrics_rec_preds[r] = [cls_metrics_preds[i]]
                            cls_metrics_rec_y_true[r] = [cls_metrics_y_true[i]]

            # store est alpha per iteration
            est_alphas.append(sum(alpha_list))

            # compute the final posterior using all clusters' predictions
            post_est = PosteriorEstimation(scar=False)
            preds, y_ml, y_orig, rec_ids, calibrated_preds = \
                post_est.compute_posterior_probs(rec_preds, rec_y_ml, rec_y_orig, cal_preds_dict=rec_cal_preds)

            # write classification results to an output file
            rec_count = Counter(y_ml)
            r_fileop.write_ml_prediction_results(rec_count[1], rec_count[0], y_ml, preds, y_orig, rec_ids, itr,
                                                 calibrated_preds, cluster='', p_frac='')

            # check if classification performance metrics need to be calculated
            if self.classification_metrics:
                # compute the final posterior using all clusters' predictions
                post_est = PosteriorEstimation(scar=False)
                cls_metrics_preds, _, cls_metrics_y_true, _, _ = \
                    post_est.compute_posterior_probs(cls_metrics_rec_preds, cls_metrics_rec_y_true,
                                                     cls_metrics_rec_y_true, cal_preds_dict=None)

                # calculate perofmance metrics
                itr_bs, itr_aps, itr_auc, itr_f1, itr_mcc, itr_acc = \
                    pulsnar_performance_metrics(cls_metrics_preds, cls_metrics_y_true, scar=False)

                # store performance metrics
                clf_bs.append(itr_bs)
                clf_aps.append(itr_aps)
                clf_auc.append(itr_auc)
                clf_f1.append(itr_f1)
                clf_mcc.append(itr_mcc)
                clf_acc.append(itr_acc)

        # return estimated alpha and output files
        res['estimated_alpha'] = np.mean(est_alphas)
        res['prediction_file'] = io_files['result_file']
        res['alpha_file'] = io_files['alpha_file']
        res['important_features_file'] = io_files['imp_feature_file']
        res['bic_vs_cluster_count_plot'] = io_files['bic_plot_file']
        if self.classification_metrics:
            res['pulsnar_brier_score'] = np.mean(clf_bs)
            res['pulsnar_average_precision_score'] = np.mean(clf_aps)
            res['pulsnar_auc_roc_score'] = np.mean(clf_auc)
            res['pulsnar_f1_score'] = np.mean(clf_f1)
            res['pulsnar_mcc_score'] = np.mean(clf_mcc)
            res['pulsnar_accuracy'] = np.mean(clf_acc)

        return res

    def run_ml_and_estimate_alpha(self, X, Y, Y_true, mv_list, itr):
        """
        Run the classifier on the given data and estimate the fraction of positives among unlabeled using predicted
        probabilities.
        """
        # get ML parameters
        classifier_params = self.get_params(option='classifier')
        # print("classifier_params: ", classifier_params)

        # instantiate the classifier and run it
        ml_clf = ClassificationEstimator(clf=self.classifier, clf_params=classifier_params)
        preds, y_ml, y_orig, rec_ids, _ = ml_clf.train_test_model(X, Y, Y_true, mv_list,
                                                                  k_folds=self.kfold,
                                                                  rseed=itr, calibration=None)

        # estimate the fraction of positives in the unlabeled set
        pfe = PositiveFractionEstimation(bin_method=self.bin_method, lowerbw=self.lowerbw, upperbw=self.upperbw,
                                         bw_method=self.bw_method, optim=self.optim)
        est_alpha = pfe.estimate_positive_fraction_in_unlabeled(preds, y_ml)
        return preds, y_ml, y_orig, rec_ids, est_alpha

    def get_params(self, option='classifier'):
        """
        Check if user provided parameters for the classifier. If not, get the default value from the
        PulsnarParams.py file
        """
        # check if param file was provided by user?
        param_file = False
        config = None
        if self.pulsnar_params_file is not None and os.path.exists(self.pulsnar_params_file):
            # read the parameters file
            with open(self.pulsnar_params_file, 'r') as fyml:
                config = yaml.safe_load(fyml)
            param_file = True

        # return the parameters depending on options.
        # classifier parameters
        if option == 'classifier':
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
        elif option == 'IO':
            if param_file:
                return dict(config['IO_params'])
            else:
                return pp.IO_params
        elif option == 'clustering':
            if param_file:
                return dict(config['GMM_params'])
            else:
                return pp.GMM_params
        else:
            logging.error("Invalid option !!!")

    def apply_calibration(self, preds, y_ml, y_orig, rec_ids, est_alpha):
        """
        Run the calibration algorithm and return calibrated probabilities
        """
        # apply calibration to PU data or only U data
        cal_probs = CalibrateProbabilities(calibration_data=self.calibration_data, n_bins=self.calibration_n_bins,
                                           alpha=est_alpha, calibration_method=self.calibration_method,
                                           smooth_isotonic=self.smooth_isotonic)

        preds, y_ml, y_orig, rec_ids, calibrated_preds = \
            cal_probs.calibrate_predicted_probabilities(preds, y_ml, y_orig, rec_ids, k_flips=self.kflips)

        return preds, y_ml, y_orig, rec_ids, calibrated_preds

    # SNAR specific functions
    def get_model_imp_features(self, data, label):
        """
        Run the model with full data to get the list of important features used in the model training.
        Save features as training model on 20m records takes ~12 hours
        """
        # get ML parameters
        classifier_params = self.get_params(option='classifier')

        # instantiate the classifier and run it
        ml_clf = ClassificationEstimator(clf=self.classifier, clf_params=classifier_params)

        # get important features
        f_idx, f_imp_vals = ml_clf.find_important_features(data, label, imp_type="gain")
        impf = dict(zip(f_idx, f_imp_vals))
        return impf

    def determine_clusters(self, imf, X_pos, covar, bic_plt_file, n_clusters=None, csr=False, top50p=False):
        """
        split labeled positive tests into k clusters
        """
        # get GMM parameters
        gmm_params = self.get_params(option='clustering')

        # set some parameters for clustering algorithm
        gmm_params['covariance_type'] = covar
        cls = ClusteringEstimator(clf="gmm", clf_params=gmm_params)

        # dict to array - feature index and feature importance
        f_idx, f_imp_vals = list(map(np.array, zip(*imf.items())))

        # check if number of clusters needs to be estimated
        if n_clusters is None:
            aic_vals, bic_vals, n_clusters = cls.find_cluster_count(X_pos, f_idx, f_imp_vals,
                                                                    max_clusters=self.max_clusters, n_iters=1,
                                                                    n_threads_blas=1, top50p=top50p, csr=csr)
            print('Number of clusters in the positive set: ', n_clusters)
            plt = MiscUtils(bic_plot=True)

            # generate BIC plot
            cluster_list = [c + 1 for c in range(len(bic_vals))]
            plt.draw_line_plot(cluster_list, bic_vals, bic_plt_file)
            # n_clusters = int(input("Check the BIC plot and enter number of cluster: "))

        # divide positives into n clusters
        cluster_indx, sel_idx = cls.divide_positives_into_clusters(X_pos, f_idx, f_imp_vals, n_clusters,
                                                                   n_threads_blas=1, top50p=top50p, csr=csr)
        return cluster_indx, sel_idx
