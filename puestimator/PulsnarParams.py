# **** depending on the classifier you want to run, comment/uncomment the classifier_params. *** #

# Parameters for XGBoost. If you want to add more parameters of XGBoost,
# check the list here: https://xgboost.readthedocs.io/en/stable/parameter.html
XGB_params = {
    'n_jobs': 16,
    'max_depth': 3,
    'scale_pos_weight': 1.0,
    'random_state': 101
}

# Parameters for LR. If you want to add more parameters of LR,
# check the list here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
LR_params = {'penalty': 'l2',
             'max_iter': 5000,
             'verbose': 0,
             'n_jobs': 16,
             'class_weight': None,
             'random_state': 101}

# Parameters for catboost. If you want to add more parameters of catboost,
# check the list here: https://catboost.ai/en/docs/references/training-parameters/
CB_params = {
    'thread_count': 16,
    'depth': 3,
    'random_seed': 1007
}

# Parameters for GMM algorithm. If you want to add more parameters of GMM,
# check the list here: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
GMM_params = {
    'n_components': 1,
    'covariance_type': 'full',
    'random_state': 101
}

# files to store IO
IO_params = {
    'result_file': 'results/predictions.tsv',
    'alpha_file': 'results/alpha_estimates.tsv',
    'imp_feature_file': 'results/model_imp_features.pkl',
    'bic_plot_file': 'plots/bic_vs_cluster_count.png'
}

# parameters for XGBoost GridSearch
xgb_grid_params = {
    'eta': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 1],
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
}

