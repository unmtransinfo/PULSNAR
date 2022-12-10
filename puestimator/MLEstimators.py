from sklearn.base import BaseEstimator
from threadpoolctl import threadpool_limits
import logging
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from scipy.signal import argrelmin
from collections import Counter
from scipy import sparse


# *** Methods for Classification ****
class ClassificationEstimator(BaseEstimator):
    """
    A wrapper for sklearn ML estimators
    """

    def __init__(self, clf=None, clf_params=None):

        if clf_params is None:
            clf_params = {'random_state': 107}

        self.clf = clf
        self.clf_params = clf_params

        self.model = self.select_the_model()
        # check if the classifier supports fit and predict_proba
        if not hasattr(self.model, "fit"):
            logging.error("The selected classifier does not have fit() function")
            exit(0)
        if not hasattr(self.model, "predict_proba"):
            logging.error("The selected classifier does not have predict_proba() function")
            exit(0)

    def select_the_model(self):
        """
        This function checks which classifier needs to be used for the model-training.
        Currently, it supports XGBoost, CatBoost, and Logistic Regression. Modify this function to include other
        classifiers. For non-SCAR data, we use feature importance to divide positive samples into clusters.
        So, use only those classifiers that return feature importance.

        Parameters
        ----------

        Returns
        -------
        chosen model
        """
        if self.clf is None or self.clf == "xgboost":
            # Use XGBoost if no classifier is provided
            model = xgb.XGBClassifier(**self.clf_params)
        elif self.clf == "lr":
            model = LogisticRegression(**self.clf_params)
        elif self.clf == "catboost":
            self.clf_params['verbose'] = False
            model = cb.CatBoostClassifier(**self.clf_params)
        else:
            logging.error("MLEstimators.py needs to be modified to support {0}".format(self.clf))
            exit(-1)
        return model

    def train_test_model(self, data, ml_label, tru_label, rec_ids, k_folds=5, rseed=7, calibration=None):
        """
        This method trains and tests the ML model using the given data and labels. In this function, the given data
        are split into train and test sets using k_folds value.

        Parameters
        ----------
        data: feature matrix for ML model
        ml_label: labels for ML model
        tru_label: true labels of the data (used only for known datasets)
        rec_ids: record ids
        k_folds: number of folds
        rseed: random seed value
        calibration: should predicted probabilities be calibrated

        Returns
        -------
        ML predicted probabilities,
        machine learning labels,
        true labels,
        record ids, and
        calibrated predicted probabilities.
        """

        # variables to store predicted probabilities, ml labels, and true labels
        preds = None
        calibrated_preds = None
        y_ml = None
        y_true = None
        recs = None

        # If true labels are unknown, pass tru_label as None`
        if tru_label is None:
            tru_label = ml_label

        # divide data and labels in k groups of train and test set
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=rseed)
        for fold, (tr_idx, te_idx) in enumerate(skf.split(data, ml_label)):
            # train the model
            fold_data_tr = data[tr_idx]
            fold_label_tr = ml_label[tr_idx]
            tot = Counter(fold_label_tr)
            if self.clf == "lr":
                self.clf_params['class_weight'] = {0: 1, 1: tot[0] / tot[1]}
            else:
                self.clf_params['scale_pos_weight'] = tot[0] / tot[1]
            # check if classifier needs to be calibrated
            if calibration is None:
                model = self.select_the_model()
                model.fit(fold_data_tr, fold_label_tr)
            else:
                clf = self.select_the_model()
                model = CalibratedClassifierCV(base_estimator=clf, cv=3, method=calibration)
                clf.fit(fold_data_tr, fold_label_tr)
                model.fit(fold_data_tr, fold_label_tr)

            # test the model and store values
            if fold == 0:
                y_ml = ml_label[te_idx]
                y_true = tru_label[te_idx]
                recs = rec_ids[te_idx]
                if calibration is None:
                    preds = model.predict_proba(data[te_idx])
                    calibrated_preds = preds
                else:
                    preds = clf.predict_proba(data[te_idx])
                    calibrated_preds = model.predict_proba(data[te_idx])
            else:
                y_ml = np.concatenate([y_ml, ml_label[te_idx]])
                y_true = np.concatenate([y_true, tru_label[te_idx]])
                recs = np.concatenate([recs, rec_ids[te_idx]])
                if calibration is None:
                    preds = np.concatenate([preds, model.predict_proba(data[te_idx])], axis=0)
                    calibrated_preds = np.concatenate([calibrated_preds, preds], axis=0)
                else:
                    preds = np.concatenate([preds, clf.predict_proba(data[te_idx])], axis=0)
                    calibrated_preds = np.concatenate([calibrated_preds, model.predict_proba(data[te_idx])], axis=0)
        return preds[:, 1], y_ml, y_true, recs, calibrated_preds[:, 1]

    def find_hyper_parameters(self, data, label, verbose=5, kfold=5, njobs=4, score='average_precision'):
        """
        This function runs gridsearch to find the optimal parameters for the given classifier.

        Parameters
        ----------
        data: feature matrix for ML model
        label: labels for ML model
        verbose: a parameter for GridSearch. Amount of details to be displayed on the terminal
        kfold: number of folds
        njobs: number of threads to use
        score: scoring parameter for the GridSearch

        Returns
        -------
        optimal parameters for the classifier
        """

        '''
        # which classifier to select for parameter tuning
        if self.clf == "xgboost":
            gs_params = cp.xgb_grid_params
        elif self.clf == "lr":
            gs_params = cp.lr_grid_params
        elif self.clf == "catboost":
            gs_params = cp.cb_grid_params
        else:
            logging.error("{0} no supported. Codes need to be modified".format(self.clf))
            exit(-1)
        '''
        # determine range for scale_pos_weight
        tot = Counter(label)
        r = int(tot[0] / tot[1]) + 1
        gs_params = self.clf_params
        gs_params['scale_pos_weight'] = [i for i in range(1, r + 1, 2)]
        # stat the grid-search. If the classifier is from scikit-package, use GridSearchCV.
        if self.clf == "xgboost" or self.clf == "lr":
            gs = GridSearchCV(self.model, n_jobs=njobs, param_grid=gs_params, scoring=score, cv=kfold, verbose=verbose)
            gs.fit(data, label)
            opt_params = gs.best_params_
        elif self.clf == "catboost":
            gs = self.model.grid_search(gs_params, data, y=label, cv=kfold)
            opt_params = gs['params']
        logging.info("Parameters used in the best model: {0}".format(opt_params))
        return opt_params

    def find_important_features(self, data, label, sep=None, imp_type=None):
        """
        This function fits a given model and returns important features and their feature importance.
        Currently, it supports XGBoost, LR, and CatBoost. Modify this function to add other classifiers.

        Parameters
        ----------
        data: feature matrix for ML model
        label: labels for ML model
        sep: used by CatBoost to generate its Pool
        imp_type: used by XGBoost e.g. gain

        Returns
        -------
        indices of important features
        feature importance values
        """
        feature_idx = None
        importance_vals = None

        # find importance scores of features
        if self.clf == "xgboost":
            tot = Counter(label)
            self.clf_params['scale_pos_weight'] = tot[0] / tot[1]  # set scale_pos_weight parameter
            model = self.select_the_model()
            bst = model.fit(data, label)
            imp_features = bst.get_booster().get_score(importance_type=imp_type)
            imp_features = Counter(imp_features).most_common()  # sort the important features by their importance
            feature_idx = np.asarray([int(f[0][1:]) for f in imp_features])
            importance_vals = np.asarray([f[1] for f in imp_features])
        elif self.clf == "catboost":
            tot = Counter(label)
            self.clf_params['scale_pos_weight'] = tot[0] / tot[1]  # set scale_pos_weight parameter
            model = self.select_the_model()
            X = cb.Pool(data=data, label=label, delimiter=sep)
            bst = model.fit(X)
            importance_vals = bst.get_feature_importance(X)
            feature_idx = np.asarray([i for i in range(len(importance_vals))])
            j = np.argsort(importance_vals)[::-1]  # index of imp vals sorted in decreasing order
            feature_idx = feature_idx[j]
            importance_vals = importance_vals[j]
        elif self.clf == "lr":
            bst = self.model.fit(data, label)
            importance_vals = np.asarray([abs(iv) for iv in bst.coef_[0]])
            feature_idx = np.asarray([i for i in range(len(importance_vals))])
            j = np.argsort(importance_vals)[::-1]  # index of imp vals sorted in decreasing order
            feature_idx = feature_idx[j]
            importance_vals = importance_vals[j]

        return feature_idx, importance_vals


# ****** Clustering functions ****
# this function is used by clustering algorithm
def preprocess_data(data, f_idx, f_imp_vals, top50p, csrdata):
    """
    This function scales the important features by their importance scores.

    Parameters
    ----------
    data: feature matrix for ML
    f_idx: indices of the important features
    f_imp_vals: importance value of the important features
    top50p: should only top50 important features be selected?
    csrdata: are data in csr format?

    Returns
    -------
    scaled features,
    feature indices
    """

    # logging.info("Shape of the data before scaling: {0}".format(data.shape))
    if top50p:
        half = int(len(f_idx) / 2)
        idx = f_idx[:half]
        importance_score = f_imp_vals[:half]
    else:
        idx = f_idx
        importance_score = f_imp_vals

    # scale important features by their scores
    if csrdata:
        sel_data = data[:, idx].toarray() * importance_score
    else:
        sel_data = data[:, idx] * importance_score

    # logging.info("Shape of the data after scaling: {0}".format(sel_data.shape))
    return sel_data, idx


def check_cluster_purity(cluster_indx, true_labels):
    """
    This function checks the purity of clusters i.e. fraction of different types of positives in
    each cluster.

    Parameters
    -----------
    cluster_indx: which cluster?
    true_labels: true labels of the data
    """

    j = 0
    for idx in cluster_indx:
        sel_labels = true_labels[idx]
        logging.info("Fraction of labels {0} in cluster {1}: {2}".format(np.unique(true_labels), j + 1,
                                                                         np.bincount(sel_labels) / len(sel_labels)))


class ClusteringEstimator(BaseEstimator):
    """
    A wrapper for sklearn Clustering algorithms
    """

    def __init__(self, clf=None, clf_params=None):

        # check which clustering algorithm needs to be used
        if clf is None or clf == "gmm":
            # Use GMM if no clf is provided
            model = GaussianMixture(**clf_params)
        else:
            logging.error("MLEstimators.py needs to be modified to support {0}".format(clf))
            exit(0)

        # check if the classifier supports fit and predict_proba
        if not hasattr(model, "fit"):
            logging.error("The selected algorithm {0} does not have fit() function".format(clf))
            exit(0)
        if not hasattr(model, "bic"):
            logging.error("The selected algorithm {0} does not have bic() function".format(clf))
            exit(0)
        if not hasattr(model, "aic"):
            logging.error("The selected algorithm {0} does not have aic() function".format(clf))
            exit(0)

        self.clf = clf
        self.clf_params = clf_params

    def find_cluster_count(self, data, f_idx, f_imp_vals, max_clusters=25, n_iters=1, n_threads_blas=1, top50p=True,
                           csr=False):
        """
        This function trains and tests the clustering models for cluster 1...max_clusters and return BIC, and AIC list.
        It also calculates the number of clusters using BIC value. It is recommended to verify the cluster count by
        looking into the BIC plot.

        Parameters
        ----------
        data: ML data
        f_idx: indices of the important features
        f_imp_vals: importance value of the important features
        max_clusters: max clusters to use in the clustering algorithm
        n_iters: number of iterations for clustering algorithm
        n_threads_blas: number of threads for blas
        top50p: select only top 50 percent of the important features?
        csr: data are in CSR format?

        Returns
        -------
        AIC value,
        BIC value,
        number of clusters
        """

        # local function
        def num_of_clusters(vals):
            """
            Use BIC score to compute the number of clusters

            Parameters
            ----------
            vals: BIC values

            Returns
            -------
            number of clusters
            """
            vals = np.asarray(vals)
            minima_bic_idx = argrelmin(vals)[0]
            if len(minima_bic_idx) > 0:     # function is not monotonically increasing or decreasing
                minima_vals = vals[minima_bic_idx]
                i = np.argmin(minima_vals)
                return minima_bic_idx[i] + 1    # cluster count starts from 1, not 0 and hence +1
            else:   # use knee point detection algorithm
                diff_list = []
                curr_val, prev_val, next_val = vals[0], vals[0], vals[0]
                for m in range(len(vals) - 1):
                    curr_val = vals[m]
                    next_val = vals[m + 1]
                    diff = prev_val + next_val - 2 * curr_val
                    prev_val = curr_val
                    diff_list.append(diff)

                # find indices of local minima in diff values
                diff_list = np.asarray(diff_list)
                minima_idx = argrelmin(diff_list)[0]  # indices of local minima

                # create a dictionary with cluster count as key and local minimum in diff_list as value
                local_min = {}
                for m in minima_idx:
                    local_min[m+1] = diff_list[m]   # cluster count starts from 1, not 0. so m+1
                local_min = Counter(local_min).most_common()  # sort local_min in decreasing order

                # find the angle for each local minimum
                angle0 = 0
                maxfound = False
                for m, v in local_min:
                    angle1 = np.arctan(1 / (abs(vals[m] - vals[m - 1]))) + np.arctan(1 / (abs(vals[m + 1] - vals[m])))
                    if angle0 == 0:
                        angle0 = angle1
                    elif angle1 > angle0:
                        maxfound = True
                        break
                    else:
                        continue
                # return cluster count. if maxima is not found, return the cluster for max local minima
                if maxfound:
                    return m
                else:
                    return local_min[0][0]

        # *** start - feature scaling *** #
        bic_values = []
        aic_values = []
        # logging.info("Scale the data feature by their importance value")
        data, _ = preprocess_data(data, f_idx, f_imp_vals, top50p, csr)

        # Run GMM clustering with updated params
        with threadpool_limits(limits=n_threads_blas, user_api='blas'):
            for n_clusters in range(1, max_clusters + 1, 1):
                self.clf_params['n_components'] = n_clusters
                temp_bic = []
                temp_aic = []
                for itr in range(n_iters):
                    # logging.info("running GMM with {0} clusters for iteration {1}".format(n_clusters, itr))
                    rv = 100 + n_clusters + itr
                    self.clf_params['random_state'] = rv
                    gm = GaussianMixture(**self.clf_params).fit(data)
                    temp_bic.append(gm.bic(data))
                    temp_aic.append(gm.aic(data))
                # add the mean value of bic and aic
                bic_values.append(np.mean(temp_bic))
                aic_values.append(np.mean(temp_aic))
        return aic_values, bic_values, num_of_clusters(bic_values)

    def divide_positives_into_clusters(self, data, f_idx, f_imp_vals, n_clusters, n_threads_blas=1, top50p=True,
                                       csr=False):
        """
        This function divides positive tests into n clusters using GMM algorithm

        Parameters
        ----------
        data: feature matrix for ML
        f_idx: indices of the important features
        f_imp_vals: importance value of the important features
        n_clusters: number of clusters
        n_threads_blas: number of threads for blas
        top50p: select only top 50 percent of the important features?
        csr: data are in CSR format?

        Returns
        -------
        indices of records in each cluster,
        indices of important features
        """

        data, sel_idx = preprocess_data(data, f_idx, f_imp_vals, top50p, csr)

        # Train GMM model for clustering
        self.clf_params['n_components'] = n_clusters
        self.clf_params['random_state'] = 100 + n_clusters

        # run GMM with selected cluster count
        with threadpool_limits(limits=n_threads_blas, user_api='blas'):
            gm = GaussianMixture(**self.clf_params).fit(data)

        # predict labels using train model
        labels = gm.predict(data)

        # group data by their labels
        cluster_indx = []
        for v in np.unique(labels):
            idx = list(np.where(labels == v)[0])
            cluster_indx.append(idx)

        return cluster_indx, sel_idx
