# PULSNAR (Positive Unlabeled Learning Selected Not At Random)

## System Requirements
The PULSNAR package has been tested on Ubuntu 22 with Python 3.10.6. The package should run on other operating systems with Python 3.7+. We used the latest version of all Python packages mentioned in the requirements.txt. Required packages are: numpy, scipy, pandas, pickle, PyYAML, scikit-learn, rpy2, xgboost, catboost, threadpoolctl.
Although we use our histogram method to select the bandwidth for the beta kernel estimator, the package supports 4 "R" methods (nrd0, nrd, bcv, and ucv) to select the bandwidth of a Gaussian kernel density estimator. nrd: normal reference distribution, bcv: biased cross-validation, and ucv: unbiased cross-validation. So, r-base functions are required to run the PULSNAR package. If r-base is not installed, it can be installed using conda.

## PULSNAR Parameters
- **scar**: data are SCAR or SNAR. If SCAR assumption holds, set the value to True, else False.
- **csrdata**: data are in csr matrix format or normal matrix format. If csr matrix, set it to True, else False
- classifier: which classifier to use. Supported classifiers are 'xgboost', 'catboost', 'lr'
- n_clusters: If number of types of positives in the data is known, set it to that value, else 0, 
- max_clusters: If number of types of positives in the data is unknown, set it to a value so that PULSNAR can determine the number of clusters. Default value is 25. Setting it to a very large value can slow down the execution of the algorithm.
- covar_type: covariance type for GMM algorithm ('full', 'diag', 'spherical', 'tied'). Default value 'full' works great.
- top50p_covars: If you want to use only top 50% of important features in determining the number of clusters, set it to True, else False
- bin_method:'scott'
- bw_method:'hist'
- lowerbw:0.01
- upperbw:0.5
- optim:'global',
- calibration:False
- calibration_data:'PU'
- calibration_method:'isotonic'
- calibration_n_bins:100
- smooth_isotonic:False
- classification_metrics:False
- n_iterations=1
- kfold=5
- kflips=1
- pulsnar_params_file=None

