from PULSNAR.pudata.SimulatedData import SklearnSimulatedData
from PULSNAR import PULSNAR
from PULSNAR.pudata import SimulatedDataParams as pc
import numpy as np
import sys
import os
from sklearn.utils import shuffle
from copy import deepcopy

# ****************************************************** #
#                   start of the code                    #
# ****************************************************** #
pfracs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]  # positive fraction
cs = 0.3
nf = 50
n_pos = 2000
n_unlab = 6000
iterations = 1
final_alpha = {}

# get parameters from user for PULSNAR algorithm
# update pulscar_synthetic_alpha.yaml if you want to override the default parameters
if len(sys.argv) < 2:
    user_param_file = 'testparams/pulscar_synthetic_alpha.yaml'
else:
    user_param_file = sys.argv[1]

# check if results folder exist. if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# generate synthetic SCAR data and run PULSCAR
for pf in pfracs:
    # instantiate simulated data class
    s_data = SklearnSimulatedData(pc.make_classification_params)

    # SCAR data
    X, Y, Y_true = s_data.generate_simulated_data(n_pos=n_pos, n_unlab=n_unlab, pf_in_unlab=pf, n_features=nf,
                                                  n_classes=2, class_sep=cs, equal_frac_in_pos=True,
                                                  discrete_features=False, scar=True, rseed=1001)

    rec_ids = np.array([i for i in range(len(Y_true))])
    # keep a copy for iterations
    orig_X, orig_Y, orig_Y_true, orig_rec_ids = deepcopy(X), deepcopy(Y), deepcopy(Y_true), deepcopy(rec_ids)

    for itr in range(iterations):
        X, Y, Y_true, rec_ids = deepcopy(orig_X), deepcopy(orig_Y), deepcopy(orig_Y_true), deepcopy(orig_rec_ids)
        # instantiate PULSNARClassifier
        pls = PULSNAR.PULSNARClassifier(scar=True, csrdata=False, classifier='xgboost',
                                        bin_method='rice', bw_method='hist', lowerbw=0.05, upperbw=0.5, optim='local',
                                        calibration=False, calibration_data='PU', calibration_method='isotonic',
                                        calibration_n_bins=100, smooth_isotonic=False,
                                        classification_metrics=False,
                                        n_iterations=1, kfold=5, kflips=1,
                                        pulsnar_params_file=user_param_file)

        # get results
        X, Y, Y_true, rec_ids = shuffle(X, Y, Y_true, rec_ids, random_state=itr)
        res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)
        final_alpha[pf, itr] = res['estimated_alpha']
        print("True alpha: {0} Estimated alpha: {1}".format(pf, res['estimated_alpha']))

# print alpha
print("\n")
print("Algorithm\tIteration\tTrue_alpha\tEst_alpha")
for k, v in final_alpha.items():
    print("PULSCAR" + "\t" + str(k[1]) + "\t" + str(k[0]) + "\t" + str(v))
