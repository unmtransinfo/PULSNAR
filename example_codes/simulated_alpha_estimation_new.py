from PULSNAR.pudata.SimulatedData import SklearnSimulatedData
from PULSNAR import PULSNAR
from PULSNAR.pudata import ParamConfig as pc
import sys

# data
pf = 0.20
n_pos = 2500
n_unlab = 2500

# get parameters from user
if len(sys.argv) < 2:
    print("PULSNAR will run with default parameters")
    user_param_file = None
else:
    user_param_file = sys.argv[1]

# instantiate simulated data class
s_data = SklearnSimulatedData(pc.make_classification_params)
'''
# SCAR data
X, Y, Y_true = s_data.generate_simulated_data(n_pos=n_pos, n_unlab=n_unlab, pf_in_unlab=pf, n_features=50,
                                              n_classes=2, class_sep=0.3, equal_frac_in_pos=True,
                                              discrete_features=False, scar=True, rseed=7)

pls = PULSNAR.PULSNARClassifier(scar=True, csrdata=False, classifier='xgboost', calibration=True,
                                calibration_data='PU', calibration_method='isotonic', smooth_isotonic=False,
                                classification_metrics=True, iterations=1, kfold=5, kflips=20,
                                pulsnar_params_file=user_param_file)

'''
# SNAR data
X, Y, Y_true = s_data.generate_simulated_data(n_pos=n_pos, n_unlab=n_unlab, pf_in_unlab=pf,
                                              n_features=50,
                                              n_classes=6, class_sep=0.3, equal_frac_in_pos=True,
                                              discrete_features=False, scar=False, rseed=7)
pls = PULSNAR.PULSNARClassifier(scar=False, csrdata=False, classifier='xgboost', n_clusters=0, covar_type='full',
                                top50p_covars=False, bin_method='scott', bw_method='hist', optim='global',
                                calibration=True,
                                calibration_data='PU', calibration_method='isotonic', smooth_isotonic=False,
                                classification_metrics=True, iterations=1, kfold=5, kflips=20,
                                pulsnar_params_file=user_param_file)


res = pls.run_pulsnar(X, Y, tru_label=Y_true)
print(res)