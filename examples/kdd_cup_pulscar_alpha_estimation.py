from PULSNAR import PULSNAR
import numpy as np
import sys
import os
import pandas as pd
from sklearn.utils import shuffle


def read_input_file(ifile):
    """
    Read the input file and return features and labels
    """
    return pd.read_csv(ifile, sep='\t', header=None)


def create_pos_unlabeled_set(data, itr=0):
    """
    Create positive and unlabeled set
    """
    np.random.seed(itr)
    data = data.values  # pandas to numpy
    labels = data[:, 1]
    dat = data[:, 2:79]
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    # select positive and unlabeled data
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
    return dat[idx1], labels[idx1], dat[idx0], labels[idx0]


# ****************************************************** #
#                   start of the code                    #
# ****************************************************** #
inpfile = "UCIdata/phy_train.dat"
pfracs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]  # positive fraction
n_iterations = 1
iter_alpha = {}

# get parameters from user for PULSNAR algorithm.
if len(sys.argv) < 2:
    user_param_file = 'testparams/kdd_alpha.yaml'
else:
    user_param_file = sys.argv[1]

# check if results folder exist. if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# fetch KDD cup data data
for pf in pfracs:
    for itr in range(n_iterations):
        data = read_input_file(inpfile)
        pos, y_true_1, unlab, y_true_0 = create_pos_unlabeled_set(data, itr=itr)

        # how many positives need to be flipped
        label_change_count = int(unlab.shape[0] * pf / (1 - pf))
        # print("pos count, unlab count, label_change_count: ", pos.shape[0], unlab.shape[0], label_change_count)

        # generate data for the PULSCAR
        X = np.vstack((pos, unlab))  # merge the data
        Y = np.concatenate([[1] * (pos.shape[0] - label_change_count), [0] * (unlab.shape[0] + label_change_count)])
        Y_true = np.concatenate([[1] * len(y_true_1), [0] * len(y_true_0)])
        rec_ids = np.array([i for i in range(len(Y_true))])
        X, Y, Y_true, rec_ids = shuffle(X, Y, Y_true, rec_ids, random_state=123 + itr)

        # instantiate PULSNARClassifier
        pls = PULSNAR.PULSNARClassifier(scar=True, csrdata=False, classifier='xgboost',
                                        bin_method='scott', bw_method='hist', lowerbw=0.01, upperbw=0.5, optim='local',
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
    print("PULSCAR" + "\t" + str(k[0]) + "\t" + str(k[1]) + "\t" + str(v))
