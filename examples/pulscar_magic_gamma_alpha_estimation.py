import numpy as np
import sys
import os
import pandas as pd
from sklearn.utils import shuffle
from PULSNAR import PULSNAR


def read_input_file(ifile):
    """
    Read the input file and return features and labels
    """
    return pd.read_csv(ifile, sep=',', header=None)


def create_ml_dataset(data, pf, itr=0):
    """
    Create positive and unlabeled set
    """
    np.random.seed(itr)
    labels = data.iloc[:, 10].to_numpy()
    # print(Counter(labels))

    # find position of positive and unlabeled examples
    idx0, idx1 = np.where(labels == 'h')[0], np.where(labels == 'g')[0]

    # positive and unlabeled sets
    Y, Y_true = np.asarray([1] * len(labels)), np.asarray([1] * len(labels))
    Y[idx0], Y_true[idx0] = 0, 0
    # print("original pos and unlab count: ", np.sum(Y), len(Y) - np.sum(Y))

    # generate data
    dat = data.iloc[:, 0:10]
    dat.replace(' ', 0, inplace=True)
    dat.replace(np.nan, 0, inplace=True)
    dat = dat.values  # dataframe to numpy

    # how many positives need to be flipped
    label_change_count = int(len(idx0) * pf / (1 - pf))
    np.random.shuffle(idx1)
    i = idx1[: label_change_count]
    # print("total label change: ", len(i), label_change_count)
    Y[i] = 0

    return dat, Y, Y_true


# ****************************************************** #
#                   start of the code                    #
# ****************************************************** #
inpfile = "UCIdata/SCAR/magic_gamma_telescope/magic04.data"
pfracs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]  # positive fraction
n_iterations = 1
iter_alpha = {}

# get parameters from user for PULSNAR algorithm.
# update pulscar_magic_gamma_alpha.yaml if you want to override the default parameters
if len(sys.argv) < 2:
    user_param_file = 'testparams/pulscar_magic_gamma_alpha.yaml'
else:
    user_param_file = sys.argv[1]

# check if results folder exist. if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# fetch magic gamma data and run PU
for pf in pfracs:
    for itr in range(n_iterations):
        data = read_input_file(inpfile)
        X, Y, Y_true = create_ml_dataset(data, pf, itr=itr)
        rec_ids = np.array([i for i in range(len(Y_true))])
        X, Y, Y_true, rec_ids = shuffle(X, Y, Y_true, rec_ids, random_state=itr)
        print("Data shape, #positive, #unlab: ", X.shape, np.sum(Y), len(Y) - np.sum(Y))

        # instantiate PULSNARClassifier
        pls = PULSNAR.PULSNARClassifier(scar=True, csrdata=False, classifier='xgboost',
                                        bin_method='rice', bw_method='hist', lowerbw=0.01, upperbw=0.5, optim='local',
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
print("results for magic gamma dataset are above")
