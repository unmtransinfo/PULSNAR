from PULSNAR.pudata.SimulatedData import SklearnSimulatedData
from PULSNAR import PULSNAR
from PULSNAR.pudata import SimulatedDataParams as pc
import numpy as np
import sys


def scar_data(n_pos, n_unlab, pf, n_features, class_sep, n_classes):
    """
    Generate synthetic data using make_classification() function
    """
    # instantiate simulated data class
    s_data = SklearnSimulatedData(pc.make_classification_params)

    # SNAR data
    X, Y, Y_true = s_data.generate_simulated_data(n_pos=n_pos, n_unlab=n_unlab, pf_in_unlab=pf, n_features=n_features,
                                                  n_classes=n_classes, class_sep=class_sep, equal_frac_in_pos=True,
                                                  discrete_features=False, scar=False, rseed=7)

    # generate record ids for ML processing
    rec_ids = np.array([i for i in range(len(Y_true))])
    return X, Y, Y_true, rec_ids


# start of the code
pf = 0.50  # fraction of positives among unlabeled
n_pos = 20000  # number of labeled positives
n_unlab = 50000  # number of unlabeled examples
n_features = 100  # number of features
class_sep = 0.3  # class separability
n_classes = 6   # unlabeled + types of positives

# generatae data
X, Y, Y_true, rec_ids = scar_data(n_pos, n_unlab, pf, n_features, class_sep, n_classes)

# get parameters from user for PULSNAR algorithm.
# update pulsnar_args.yaml if you want to override the default parameters
if len(sys.argv) < 2:
    print("PULSNAR will run with default parameters")
    user_param_file = None
else:
    user_param_file = sys.argv[1]

# instantiate PULSNARClassifier
pls = PULSNAR.PULSNARClassifier(scar=False, csrdata=False, classifier='xgboost',
                                n_clusters=5, max_clusters=25, covar_type='full', top50p_covars=False,
                                bin_method='scott', bw_method='hist', lowerbw=0.01, upperbw=0.5, optim='local',
                                calibration=True, calibration_data='PU', calibration_method='isotonic',
                                calibration_n_bins=100, smooth_isotonic=False,
                                classification_metrics=True,
                                n_iterations=1, kfold=5, kflips=1,
                                pulsnar_params_file=user_param_file)

# get results
res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)
res['true_alpha'] = pf
print(res)