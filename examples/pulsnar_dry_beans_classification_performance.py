from PULSNAR import PULSNAR
import sys
import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from collections import Counter
from sklearn.metrics import brier_score_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from PULSNAR import PULSNAR
from PULSNAR.puestimator.MLEstimators import ClassificationEstimator
import yaml


def read_input_file(ifile):
    """
    Read the input file and return features and labels
    """
    return pd.read_excel(ifile)


def create_ml_dataset(data, frac, itr=0):
    """
    Create positive and unlabeled set
    """
    unlab_class = 'DERMASON'
    np.random.seed(itr)
    labels = data['Class'].to_numpy()
    # target_names = list(Counter(labels).keys())
    # print(Counter(labels))

    # positive and unlabeled sets
    Y, Y_true = np.asarray([1] * len(labels)), np.asarray([1] * len(labels))
    idx0 = np.where(labels == unlab_class)[0]
    Y[idx0], Y_true[idx0] = 0, 0
    # print("original pos and unlab count: ", np.sum(Y), len(Y) - np.sum(Y))

    # generate data
    dat = data.drop(columns=['Class'])
    dat = dat.values  # pandas to numpy

    # flip labels of positive examples
    label_change_count = int(len(idx0) * frac / (1 - frac)) + 1
    labels_per_class = {'SIRA': int(label_change_count * 32 / 63), 'SEKER': int(label_change_count * 16 / 63),
                        'HOROZ': int(label_change_count * 8 / 63), 'CALI': int(label_change_count * 4 / 63),
                        'BARBUNYA': int(label_change_count * 2 / 63), 'BOMBAY': int(label_change_count * 1 / 63)}
    # print("total label change and per class: ", label_change_count, labels_per_class)

    for l, v in labels_per_class.items():
        idx = np.where(labels == l)[0]
        np.random.shuffle(idx)
        i = idx[: v]
        # print("change labels of class, and total change: ", l, len(i))
        Y[i] = 0

    return dat, Y, Y_true


# ****************************************************** #
#                   start of the code                    #
# ****************************************************** #
inpfile = "UCIdata/SNAR/dry_beans/Dry_Bean_Dataset.xlsx"
pfracs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]  # positive fraction
n_iterations = 1
iter_alpha = {}

# get parameters from user for PULSNAR algorithm
# update pulsnar_dry_beans_alpha.yaml if you want to override the default parameters
if len(sys.argv) < 2:
    user_param_file = 'testparams/pulsnar_dry_beans_alpha.yaml'
else:
    user_param_file = sys.argv[1]

# get XGBoost params to run it without PULSNAR
with open(user_param_file, 'r') as fyml:
    config = yaml.safe_load(fyml)

# check if results folder exist. if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# output file to save classification metrics
rfile = open("results/snar_dry_beans_classification_metrics.tsv", 'w')
rfile.write("iteration\t" + "true_alpha\t" + "estimated_alpha\t" + "before_AUC\t" + "after_AUC\t" + "before_APS\t" +
            "after_APS\t" + "before_BS\t" + "after_BS\t" + "before_f1\t" + "after_f1\t" + "before_mcc\t" +
            "after_mcc\t" + "before_accuracy\t" + "after_accuracy\n")

# fetch data and run PU
for pf in pfracs:
    for itr in range(n_iterations):
        data = read_input_file(inpfile)
        X, Y, Y_true = create_ml_dataset(data, pf, itr=itr)
        rec_ids = np.array([i for i in range(len(Y_true))])
        X, Y, Y_true, rec_ids = shuffle(X, Y, Y_true, rec_ids, random_state=itr)
        print("Data shape, #positive, #unlab: ", X.shape, np.sum(Y), len(Y) - np.sum(Y))

        # run ML to get predictions without applying PULSNAR
        ml_clf = ClassificationEstimator(clf="xgboost", clf_params=dict(config['XGB_params']))
        preds, y_ml, y_orig, rec_ids, _ = ml_clf.train_test_model(X, Y, Y_true, rec_ids, k_folds=5,
                                                                  rseed=itr, calibration=None)
        # classification metrics before calibration
        before_bs = brier_score_loss(y_orig, preds)
        before_aps = average_precision_score(y_orig, preds)
        before_auc = roc_auc_score(y_orig, preds)
        before_f1 = f1_score(y_orig, np.round(preds).astype(int))
        before_mcc = matthews_corrcoef(y_orig, np.round(preds).astype(int))
        before_acc = accuracy_score(y_orig, np.round(preds).astype(int))

        # instantiate PULSNARClassifier
        pls = PULSNAR.PULSNARClassifier(scar=False, csrdata=False, classifier='xgboost',
                                        n_clusters=0, max_clusters=15, covar_type='full', top50p_covars=True,
                                        bin_method='rice', bw_method='hist', lowerbw=0.05, upperbw=0.5, optim='local',
                                        calibration=False, calibration_data='PU', calibration_method='isotonic',
                                        calibration_n_bins=100, smooth_isotonic=False,
                                        classification_metrics=True,
                                        n_iterations=1, kfold=5, kflips=1,
                                        pulsnar_params_file=user_param_file)
        # get results
        res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)
        iter_alpha[itr, pf] = res['estimated_alpha']
        print("True alpha: {0} Estimated alpha: {1}".format(pf, res['estimated_alpha']))

        # write metrics to the file
        line = str(itr) + "\t" + str(pf) + "\t" + str(res['estimated_alpha']) + "\t" + str(before_auc) + "\t" + \
               str(res['pulsnar_auc_roc_score']) + "\t" + str(before_aps) + "\t" + \
               str(res['pulsnar_average_precision_score']) + "\t" + str(before_bs) + "\t" + \
               str(res['pulsnar_brier_score']) + "\t" + str(before_f1) + "\t" + str(res['pulsnar_f1_score']) + "\t" + \
               str(before_mcc) + "\t" + str(res['pulsnar_mcc_score']) + "\t" + str(before_acc) + "\t" + \
               str(res['pulsnar_accuracy']) + "\n"
        rfile.write(line)

# print alpha
print("\n")
print("Algorithm\tIteration\tTrue_alpha\tEst_alpha")
for k, v in iter_alpha.items():
    print("PULSNAR" + "\t" + str(k[0]) + "\t" + str(k[1]) + "\t" + str(v))
print("results for dry beans dataset are above")
