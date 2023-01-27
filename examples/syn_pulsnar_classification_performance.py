from PULSNAR.pudata.SimulatedData import SklearnSimulatedData
from PULSNAR import PULSNAR
from PULSNAR.pudata import SimulatedDataParams as pc
from PULSNAR.puestimator.MLEstimators import ClassificationEstimator
import numpy as np
import sys
import os
import yaml
from sklearn.utils import shuffle
from sklearn.metrics import brier_score_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

# ****************************************************** #
#                   start of the code                    #
# ****************************************************** #
pfracs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]  # positive fraction
cs = 0.3
nf = 100
n_pos = 20000
n_unlab = 50000
iterations = 1

# get parameters from user for PULSNAR algorithm.
if len(sys.argv) < 2:
    user_param_file = 'testparams/classification_performance.yaml'
else:
    user_param_file = sys.argv[1]

# get XGBoost params to run it without PULSNAR
with open(user_param_file, 'r') as fyml:
    config = yaml.safe_load(fyml)

# check if results folder exist. if not, create it
if not os.path.exists("results"):
    os.makedirs("results")

# output file to save classification metrics
rfile = open("results/snar_syn_classification_metrics1.tsv", 'w')
rfile.write("iteration\t" + "true_alpha\t" + "estimated_alpha\t" + "before_AUC\t" + "after_AUC\t" + "before_APS\t" +
            "after_APS\t" + "before_BS\t" + "after_BS\t" + "before_f1\t" + "after_f1\t" + "before_mcc\t" +
            "after_mcc\t" + "before_accuracy\t" + "after_accuracy\n")

# create data and get results
for pf in pfracs:
    # instantiate simulated data class
    s_data = SklearnSimulatedData(pc.make_classification_params)

    # SNAR data
    X, Y, Y_true = s_data.generate_simulated_data(n_pos=n_pos, n_unlab=n_unlab, pf_in_unlab=pf,
                                                  n_features=nf, n_classes=6, class_sep=cs,
                                                  equal_frac_in_pos=True, discrete_features=False,
                                                  scar=False, rseed=7)
    rec_ids = np.array([i for i in range(len(Y_true))])

    for itr in range(iterations):
        # shuffle data before calling classifier/PULSNAR
        X, Y, Y_true, rec_ids = shuffle(X, Y, Y_true, rec_ids, random_state=itr)

        # run ML to get predictions without applying PULSNAR
        ml_clf = ClassificationEstimator(clf="xgboost", clf_params=dict(config['XGB_params']))
        preds, y_ml, y_orig, rec_ids, _ = ml_clf.train_test_model(X, Y, Y_true, rec_ids, k_folds=5,
                                                                  rseed=itr, calibration=None)

        # classification metrics before calibration
        y_orig[y_orig > 1] = 1
        before_bs = brier_score_loss(y_orig, preds)
        before_aps = average_precision_score(y_orig, preds)
        before_auc = roc_auc_score(y_orig, preds)
        before_f1 = f1_score(y_orig, np.round(preds).astype(int))
        before_mcc = matthews_corrcoef(y_orig, np.round(preds).astype(int))
        before_acc = accuracy_score(y_orig, np.round(preds).astype(int))

        # instantiate PULSNARClassifier
        pls = PULSNAR.PULSNARClassifier(scar=False, csrdata=False, classifier='xgboost',
                                        n_clusters=5, max_clusters=25, covar_type='full', top50p_covars=False,
                                        bin_method='scott', bw_method='hist', lowerbw=0.01, upperbw=0.5, optim='local',
                                        calibration=False, calibration_data='PU', calibration_method='isotonic',
                                        calibration_n_bins=100, smooth_isotonic=False,
                                        classification_metrics=True,
                                        n_iterations=1, kfold=5, kflips=100,
                                        pulsnar_params_file=user_param_file)
        # get PULSNAR results
        res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)

        # write metrics to the file
        line = str(itr) + "\t" + str(pf) + "\t" + str(res['estimated_alpha']) + "\t" + str(before_auc) + "\t" + \
               str(res['pulsnar_auc_roc_score']) + "\t" + str(before_aps) + "\t" + \
               str(res['pulsnar_average_precision_score']) + "\t" + str(before_bs) + "\t" + \
               str(res['pulsnar_brier_score']) + "\t" + str(before_f1) + "\t" + str(res['pulsnar_f1_score']) + "\t" + \
               str(before_mcc) + "\t" + str(res['pulsnar_mcc_score']) + "\t" + str(before_acc) + "\t" + \
               str(res['pulsnar_accuracy']) + "\n"
        rfile.write(line)

# close output file
rfile.close()
