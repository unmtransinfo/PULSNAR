from PULSNAR import PULSNAR
import sys
import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from collections import Counter


def read_input_file(ifile):
    """
    Read the input file and return features and labels
    """
    return pd.read_csv(ifile, sep=',', header=0)


def create_ml_dataset(data, frac, itr=0):
    """
    Create positive and unlabeled set
    """
    data = data[data['Family'] != 'Bufonidae'].reset_index()
    np.random.seed(itr)
    unlab_class = 'Hylidae'
    labels = data['Family']
    p_idx = np.where(labels != unlab_class)[0]
    p_labels = labels[p_idx]
    # print(Counter(labels), len(Counter(labels)))

    # positive and unlabeled sets
    Y, Y_true = np.asarray([1] * len(labels)), np.asarray([1] * len(labels))
    idx0 = np.where(labels == unlab_class)[0]
    Y[idx0], Y_true[idx0] = 0, 0
    # print("original pos and unlab count: ", np.sum(Y), len(Y) - np.sum(Y))

    # generate data
    dat = data.drop(columns=['Family', 'Genus', 'Species', 'RecordID'])
    dat.replace(' ', 0, inplace=True)
    dat.replace(np.nan, 0, inplace=True)
    dat = dat.values  # pandas to numpy

    # flip labels of positive examples - unlabeled will have equal number of positives of each class, positives will have different ratio
    label_change_count = int(len(idx0) * frac / (1 - frac)) + 1
    k = len(Counter(labels)) - 1    # number of positive subtypes
    labels_per_class = {vv[0]: int(round(label_change_count * (2**j/(2**k - 1)))) for j, vv in enumerate(Counter(p_labels).most_common()[::-1])}
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
inpfile = "UCIdata/SNAR/anuran/Frogs_MFCCs.csv"
pfracs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]  # positive fraction
n_iterations = 1
iter_alpha = {}

# get parameters from user for PULSNAR algorithm
# update pulsnar_anuran_alpha.yaml if you want to override the default parameters
if len(sys.argv) < 2:
    user_param_file = 'testparams/pulsnar_anuran_alpha.yaml'
else:
    user_param_file = sys.argv[1]

# check if results folder exist. if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# fetch anuran data and run PU
for pf in pfracs:
    for itr in range(n_iterations):
        data = read_input_file(inpfile)
        X, Y, Y_true = create_ml_dataset(data, pf, itr=itr)
        rec_ids = np.array([i for i in range(len(Y_true))])
        X, Y, Y_true, rec_ids = shuffle(X, Y, Y_true, rec_ids, random_state=itr)
        print("Data shape, #positive, #unlab: ", X.shape, np.sum(Y), len(Y) - np.sum(Y))

        # instantiate PULSNARClassifier
        pls = PULSNAR.PULSNARClassifier(scar=False, csrdata=False, classifier='xgboost',
                                        n_clusters=0, max_clusters=15, covar_type='full', top50p_covars=True,
                                        bin_method='rice', bw_method='hist', lowerbw=0.05, upperbw=0.5, optim='local',
                                        calibration=False, calibration_data='PU', calibration_method='isotonic',
                                        calibration_n_bins=100, smooth_isotonic=False,
                                        classification_metrics=False,
                                        n_iterations=1, kfold=5, kflips=1,
                                        pulsnar_params_file=user_param_file)
        # get results
        res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)
        iter_alpha[itr, pf] = res['estimated_alpha']
        print("True alpha: {0} Estimated alpha: {1}".format(pf, res['estimated_alpha']))

# print alpha
print("\n")
print("Algorithm\tIteration\tTrue_alpha\tEst_alpha")
for k, v in iter_alpha.items():
    print("PULSNAR" + "\t" + str(k[0]) + "\t" + str(k[1]) + "\t" + str(v))
print("results for anuran dataset are above")
