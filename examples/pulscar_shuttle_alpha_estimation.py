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
    return pd.read_csv(ifile, sep=' ', header=None)


def create_ml_dataset(data, frac, itr=0):
    """
    Create positive and unlabeled set
    """
    np.random.seed(itr)
    data = data.values  # pandas to numpy
    labels = data[:, 9]
    dat = data[:, :9]
    per_class_rec_count = Counter(labels)

    # positive and unlabeled sets
    idx0 = np.where(labels == 1)[0]
    Y, Y_true = np.asarray([1] * len(labels)), np.asarray([1] * len(labels))
    Y[idx0], Y_true[idx0] = 0, 0
    # print("Before flipping: ", per_class_rec_count, Counter(Y))

    # flip labels of positive examples
    label_change_count = int(len(idx0) * frac / (1 - frac))
    # print("labels to flip (total): ", label_change_count)

    # for smaller class, randomly select positives to flip
    for kk, vv in per_class_rec_count.items():
        if kk == 1 or kk == 4 or kk == 5:  # 1 is unlabeled, 4 and 5 are bigger positive classes
            continue
        else:
            idx = np.where(labels == kk)[0]
            np.random.shuffle(idx)
            to_flip = np.random.randint(vv * frac, vv)
            label_change_count = label_change_count - to_flip
            i = idx[: to_flip]
            # print("flipping {0} records from label {1}".format(to_flip, kk))
            Y[i] = 0
            # print("After flipping: ", Counter(Y))

    # for bigger class, keep equal number of different types of positives in the positive/unlabeled set
    # suppose x records from class 4 and 5 are not flipped, then (rec_count_4 - x ) + (rec_count_5 -x) = remaining flipping count
    # class 4
    # print("remaining labels to flip: ", label_change_count)
    idx4 = np.where(labels == 4)[0]
    idx5 = np.where(labels == 5)[0]
    # rec count not to be flipped
    not_to_flip = len(idx4) + len(idx5) - label_change_count
    not_to_flip4 = int(not_to_flip / 2)
    not_to_flip5 = not_to_flip - not_to_flip4

    # check number of records not to flip > total records in that class.... in this case, keep equal number of records in the unlabeled set
    if not_to_flip4 > len(idx4) or not_to_flip5 > len(idx5):
        # print("euqal positive in unlabeled set")
        equal_flip = int(label_change_count / 2)
        not_to_flip4 = len(idx4) - equal_flip
        not_to_flip5 = len(idx5) - (label_change_count - equal_flip)

    # rec count to be flipped
    to_flip4 = len(idx4) - not_to_flip4
    to_flip5 = len(idx5) - not_to_flip5
    # print("rec 4 flip count, no flip count: ", to_flip4, not_to_flip4)
    # print("rec 5 flip count, no flip count: ", to_flip5, not_to_flip5)

    # flip class 4 and 5 records
    # class 4
    np.random.shuffle(idx4)
    i = idx4[: to_flip4]
    # print("class 4 flip count: ", len(i), np.sum(Y[i]))
    Y[i] = 0
    # print("After flipping: ", Counter(Y))

    # class 5
    np.random.shuffle(idx5)
    i = idx5[: to_flip5]
    # print("class 5 flip count: ", len(i), np.sum(Y[i]))
    Y[i] = 0
    # print("After flipping: ", Counter(Y))
    return dat, Y, Y_true


# ****************************************************** #
#                   start of the code                    #
# ****************************************************** #
inpfile = "UCIdata/SNAR/shuttle/shuttle.data"
pfracs = [0.01, 0.05, 0.10, 0.15]  # positive fraction
n_iterations = 1
iter_alpha = {}

# get parameters from user for PULSNAR algorithm.
# update pulscar_shuttle_alpha.yaml if you want to override the default parameters
if len(sys.argv) < 2:
    user_param_file = 'testparams/pulscar_shuttle_alpha.yaml'
else:
    user_param_file = sys.argv[1]

# check if results folder exist. if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# fetch Shuttle data and run PU
for pf in pfracs:
    for itr in range(n_iterations):
        data = read_input_file(inpfile)
        X, Y, Y_true = create_ml_dataset(data, pf, itr=itr)
        rec_ids = np.array([i for i in range(len(Y_true))])
        X, Y, Y_true, rec_ids = shuffle(X, Y, Y_true, rec_ids, random_state=itr)
        print("Data shape, #positive, #unlab: ", X.shape, np.sum(Y), len(Y) - np.sum(Y))

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
        iter_alpha[itr, pf] = res['estimated_alpha']
        print("True alpha: {0} Estimated alpha: {1}".format(pf, res['estimated_alpha']))

# print alpha
print("\n")
print("Algorithm\tIteration\tTrue_alpha\tEst_alpha")
for k, v in iter_alpha.items():
    print("PULSCAR" + "\t" + str(k[0]) + "\t" + str(k[1]) + "\t" + str(v))
print("results for shuttle dataset are above")
