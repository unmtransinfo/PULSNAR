from PULSNAR import PULSNAR
import numpy as np
import sys
import os
import pandas as pd
from sklearn.utils import shuffle
from collections import Counter


def read_input_file(ifile1, ifile2):
    """
    Read the input file and return features and labels
    """
    df1 = pd.read_csv(ifile1, sep=' ', header=None)
    df2 = pd.read_csv(ifile2, sep=' ', header=None)
    df = pd.concat([df1, df2], axis=1, ignore_index=True)
    return df


def create_ml_dataset(data, frac, itr=0):
    """
    Create positive and unlabeled set
    """
    unlab_class = 5
    np.random.seed(itr)

    # generate data and labels
    data = data.values
    labels = data[:, 561]
    # print(data.shape)
    # print(Counter(labels))
    ix = np.where(labels <= 6)[0]  # drop classes with smaller number of samples
    data = data[ix, :]
    labels = data[:, 561]
    dat = data[:, 0:561]
    # print(dat.shape)
    # print(Counter(labels))

    # positive and unlabeled sets
    Y, Y_true = np.asarray([1] * len(labels)), np.asarray([1] * len(labels))
    idx0 = np.where(labels == unlab_class)[0]
    Y[idx0], Y_true[idx0] = 0, 0
    # print("original pos and unlab count: ", np.sum(Y), len(Y) - np.sum(Y))

    # flip labels of positive examples
    label_change_count = int(len(idx0) * frac / (1 - frac)) + 1
    labels_per_class = {6: int(label_change_count * 16 / 31), 4: int(label_change_count * 8 / 31), 1: int(label_change_count * 4 / 31),
                        2: int(label_change_count * 2 / 31), 3: int(label_change_count * 1 / 31)}
    # print("total label change and per class: ", label_change_count, labels_per_class)

    for l, v in labels_per_class.items():
        idx = np.where(labels == l)[0]
        np.random.shuffle(idx)
        i = idx[: v]
        Y[i] = 0

    return dat, Y, Y_true


# ****************************************************** #
#                   start of the code                    #
# ****************************************************** #
inpfile1 = "UCIdata/SNAR/smartphone/X_train.txt"
inpfile2 = "UCIdata/SNAR/smartphone/y_train.txt"
pfracs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]  # positive fraction
n_iterations = 1
iter_alpha = {}

# get parameters from user for PULSNAR algorithm.
# update pulscar_smartphone_alpha.yaml if you want to override the default parameters
if len(sys.argv) < 2:
    user_param_file = 'testparams/pulscar_smartphone_alpha.yaml'
else:
    user_param_file = sys.argv[1]

# check if results folder exist. if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# fetch smartphone data and run PU
for pf in pfracs:
    for itr in range(n_iterations):
        data = read_input_file(inpfile1, inpfile2)
        X, Y, Y_true = create_ml_dataset(data, pf, itr=itr)
        rec_ids = np.array([i for i in range(len(Y_true))])
        X, Y, Y_true, rec_ids = shuffle(X, Y, Y_true, rec_ids, random_state=itr)
        print("Data shape, #positive, #unlab: ", X.shape, np.sum(Y), len(Y) - np.sum(Y))

        # instantiate PULSNARClassifier
        pls = PULSNAR.PULSNARClassifier(scar=True, csrdata=False, classifier='xgboost',
                                        bin_method='rice', bw_method='hist', lowerbw=0.1, upperbw=0.5, optim='local',
                                        calibration=False, calibration_data='PU', calibration_method='isotonic',
                                        calibration_n_bins=100, smooth_isotonic=False,
                                        classification_metrics=False, n_iterations=1, kfold=5, kflips=1,
                                        pulsnar_params_file=user_param_file)

        # get results
        res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)
        iter_alpha[itr, pf] = res['estimated_alpha']
        print("True alpha: {0} Estimated alpha: {1}".format(pf, res['estimated_alpha']))

# print alpha
print("\n")
print("Algorithm\tIteration\tTrue_alpha\tEst_alpha")
for k, v in iter_alpha.items():
    print("PULSCAR" + "\t" + str(k[0]) + "\t" + str(k[1]) + "\t" + str(v))
print("results for smartphone dataset are above")