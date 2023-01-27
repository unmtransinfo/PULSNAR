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
    np.random.seed(itr)
    data = data.values  # pandas to numpy
    labels = data[:, 11]
    dat = data[:, :11]
    per_class_rec_count = Counter(labels)

    # positive and unlabeled sets
    idx0 = np.where(labels == 'allow')[0]
    Y, Y_true = np.asarray([1] * len(labels)), np.asarray([1] * len(labels))
    Y[idx0], Y_true[idx0] = 0, 0
    # print("Before flipping: ", per_class_rec_count, Counter(Y))

    # flip labels of positive examples
    label_change_count = int(len(idx0) * frac / (1 - frac))
    # print("total flip: ", label_change_count)
    idx = np.where(labels == 'reset-both')[0]
    np.random.shuffle(idx)
    to_flip = np.random.randint(len(idx) * frac, len(idx))
    # print('reset-both flip: ', to_flip)
    label_change_count = label_change_count - to_flip
    i = idx[: to_flip]
    Y[i] = 0

    idx = np.where(labels == 'deny')[0]
    np.random.shuffle(idx)
    to_flip = int(label_change_count / 2)
    # print('deny flip: ', to_flip)
    i = idx[: to_flip]
    Y[i] = 0

    idx = np.where(labels == 'drop')[0]
    np.random.shuffle(idx)
    to_flip = label_change_count - to_flip
    # print('drop flip: ', to_flip)
    i = idx[: to_flip]
    Y[i] = 0

    # print("After flipping: ", Counter(Y))
    return dat, Y, Y_true


# ****************************************************** #
#                   start of the code                    #
# ****************************************************** #
inpfile = "UCIdata/firewall.csv"
pfracs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]  # positive fraction
n_iterations = 1
iter_alpha = {}

# get parameters from user for PULSNAR algorithm.
if len(sys.argv) < 2:
    user_param_file = 'testparams/firewall_alpha.yaml'
else:
    user_param_file = sys.argv[1]

# check if results folder exist. if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# fetch firewall data
for pf in pfracs:
    for itr in range(n_iterations):
        data = read_input_file(inpfile)
        X, Y, Y_true = create_ml_dataset(data, pf, itr=itr)
        rec_ids = np.array([i for i in range(len(Y_true))])
        X, Y, Y_true, rec_ids = shuffle(X, Y, Y_true, rec_ids, random_state=123 + itr)

        # instantiate PULSNARClassifier
        pls = PULSNAR.PULSNARClassifier(scar=True, csrdata=False, classifier='xgboost',
                                        bin_method='rice', bw_method='hist', lowerbw=0.02, upperbw=0.5, optim='local',
                                        calibration=False, calibration_data='PU', calibration_method='isotonic',
                                        calibration_n_bins=100, smooth_isotonic=False,
                                        classification_metrics=False,
                                        n_iterations=1, kfold=5, kflips=1,
                                        pulsnar_params_file=user_param_file)

        # get results
        res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)
        iter_alpha[pf, itr] = res['estimated_alpha']
        print("True alpha: {0} Estimated alpha: {1}".format(pf, res['estimated_alpha']))

# print alpha
print("\n")
print("Algorithm\tIteration\tTrue_alpha\tEst_alpha")
for k, v in iter_alpha.items():
    print("PULSCAR" + "\t" + str(k[0]) + "\t" + str(k[1]) + "\t" + str(v))
