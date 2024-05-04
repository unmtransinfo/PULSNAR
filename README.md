# PULSNAR (Positive Unlabeled Learning Selected Not At Random)

## About PULSNAR

Consider the following topical example: if one had a set of Twitter users that were confirmed bots, an unlabeled set of users comprising a mixture of bots and non-bots, and a set of covariates for both sets, PULSNAR can provide an estimate of the fraction of bots (and thus human users as well). Our motivation for developing PULSNAR was more on biomedical problems such as estimating the fraction of patients with an undiagnosed (or uncoded) medical condition.

PULSNAR is a state-of-the-art positive unlabeled learning algorithm that, given a set of positive examples of a class and a set of unlabeled examples comprising a mix of positives and negatives, along with covariates for both sets, one can estimate the fraction, α, of positives among the unlabeled. Further, it seeks to generate well-calibrated estimates of the probability that each unlabeled example is a positive -- even when the labeled positives are not representative of the unlabeled positives -- that is, when the so-called SCAR assumption does not hold. These well-calibrated probabilities can help with improving classification performance and allowing screening efforts with a given sensitivity and specificity.

This source code implements the positive unlabeled learning techniques described in the paper:
Praveen Kumar, Christophe G. Lambert. PULSNAR -- Positive unlabeled learning selected not at random: class proportion estimation when the SCAR assumption does not hold. [https://arxiv.org/abs/2303.08269](https://arxiv.org/abs/2303.08269). The Abstract of the paper is as follows:

Positive and Unlabeled (PU) learning is a type of semi-supervised binary classification where the machine learning algorithm differentiates between a set of positive instances (labeled) and a set of both positive and negative instances (unlabeled). PU learning has broad applications in settings where confirmed negatives are unavailable or difficult to obtain, and there is value in discovering positives among the unlabeled (e.g., viable drugs among untested compounds). Most PU learning algorithms make the selected completely at random (SCAR) assumption, namely that positives are selected independently of their features. However, in many real-world applications, such as healthcare, positives are not SCAR (e.g., severe cases are more likely to be diagnosed), leading to a poor estimate of the proportion, α, of positives among unlabeled examples and poor model calibration, resulting in an uncertain decision threshold for selecting positives. PU learning algorithms can estimate α or the probability of an individual unlabeled instance being positive or both. We propose two PU learning algorithms to estimate α, calculate calibrated probabilities for PU instances, and improve classification metrics: i) PULSCAR (positive unlabeled learning selected completely at random), and ii) PULSNAR (positive unlabeled learning selected not at random). PULSNAR uses a divide-and-conquer approach that creates and solves several SCAR-like sub-problems using PULSCAR. In our experiments, PULSNAR outperformed state-of-the-art approaches on both synthetic and real-world benchmark datasets.


## System Requirements
The PULSNAR package has been tested on *Ubuntu 22* with *Python 3.10.6*. The package should run on other operating systems with Python 3.7+. We have used the latest version of all Python packages. Required packages are: *numpy, scipy, pandas, pickle, PyYAML, scikit-learn, rpy2, xgboost, catboost, threadpoolctl.*
Although we use our histogram method to select the bandwidth for the beta kernel estimator, the package supports 4 "R" methods (*nrd0, nrd, bcv*, and *ucv*) to select the bandwidth of a Gaussian kernel density estimator. **nrd**: normal reference distribution, **bcv**: biased cross-validation, and **ucv**: unbiased cross-validation. So, r-base functions are required to run the PULSNAR package. If r-base is not installed, it can be installed using conda.

## PULSNAR Parameters
- **scar**: If SCAR assumption holds, set it to True, else False.
- **csrdata**: If features are in csr matrix format, set it to True, else False.
- **classifier**: Which classifier to use? Supported classifiers are 'xgboost', 'catboost', and 'lr' (Logistic Regression).
- **n_clusters**: If the number of types of positives in the data is known, set it to that value, else 0.
- **max_clusters**: If the number of types of positives in the data is unknown, set it to a value so that PULSNAR can determine the number of clusters. The default value is 25. Setting it to a very large value can slow down the execution of the algorithm.
- **covar_type**: covariance type ('full', 'diag', 'spherical', 'tied') for the GMM algorithm for clustering. The default value 'full' works great.
- **top50p_covars**: If you want to use only top 50% of important features in determining the number of clusters in the labeled positive set, set it to True, else False.
- **bin_method**: Method to select the number of bins for histogram and beta kernel estimator. Supported values are 'scott', 'fd', 'square-root', 'rice', and 'sturges'.
- **bw_method**: Method to compute the bandwidth for the beta kernel estimator. Supported values are 'hist', 'nrd0', 'nrd', 'ucv', and 'bcv'.
- **lowerbw**: Lower range of the bandwidth for the optimizer.
- **upperbw**: Upper range of the bandwidth for the optimizer.
- **optim**: Optimization method for determining bandwidth. Supported values are 'global', and 'local'. 'global' method is slow on larger datasets. So, use 'local' for larger datasets and 'global' for smaller datasets.
- **calibration**: If you want to calibrate the ML predicted probabilities, set it to True, else False.
- **calibration_data**: If you want to calibrate the ML predicted probabilities of both positive and unlabeled examples, set it to 'PU'. If you want to calibrate the ML predicted probabilities of only unlabeled examples, set it to 'U'.
- **calibration_method**: Method to calibrate the ML predicted probabilities. Supported values are 'isotonic', and 'sigmoid'.
- **calibration_n_bins**: Number of histogram bins used in the calibration algorithm.
- **smooth_isotonic**: By default, 'isotonic' calibration does not return smooth probabilities. Set it to True if you want to apply CubicSpline to isotonically calibrated probabilities, else False.
- **classification_metrics**: If you want to compute the classification performance metrics (e.g. AUC, Accuracy, F1, etc.), set it to True, else False.
- **n_iterations**: How many times you want to repeat the PULSNAR algorithm? Needed for calculating k% confidence interval. The default value is 1.
- **kfold**: Number of folds for ML classifier. Default is 5.
- **kflips**: How many times you want to repeat the method to determine probable positives among unlabeled examples.
- **pulsnar_params_file**: File to pass the parameters for classifiers, clustering algorithm, and setting output files. It should be a ".yml" or ".yaml" file. "pulsnar_args.yaml" in the "tests" folder is an example. The file should be passed as a command line argument. If this file is not passed, the default parameters are used for the classifier.

## How to install r-base and the PULSNAR package
Steps to install r-base and the PULSNAR package:

- Run the following command to download the latest Miniconda3 from anaconda.com

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

- Run the following command to install miniconda3

```
bash Miniconda3-latest-Linux-x86_64.sh
```

- Activate miniconda environment by running the following command

```
source YOUR_MINICONDA_INSTALLATION_FOLDER/bin/activate
```

- Run the following command to install r-base

``` 
conda install -c conda-forge r-base
```

- Clone this repository

```
git clone git@github.com:unmtransinfo/PULSNAR.git 
```

- change directory to PULSNAR

```
cd PULSNAR
```

- Install all required packages. 'pip install' does not work for rpy2, so use 'conda install'.

```
pip install -r requirements.txt
conda install conda-forge::rpy2
```

- Install PULSNAR package using ".whl" built distribution file:

```
pip install install/PULSNAR-0.0.0-py3-none-any.whl
```

## Codes and results for the paper
The "PULSNAR_paper_results.xlsx" file contains the alpha values using PULSCAR and PULSNAR on all synthetic and machine learning benchmark datasets that we used for the paper. Additionally, we have provided all the codes used for alpha estimation by PULSCAR and PULSNAR in the "examples" folder. These codes are designed to call the PULSNAR package, with the parameters set according to the data type, either SCAR or SNAR.

## How to run the PULSNAR package
Since the PULSNAR package and all required packages were installed under the conda environment, you need to activate the conda environment to run the PULSNAR. The "examples" folder contains examples showing how to run the PULSNAR package. It needs features, ML labels, fake/real record ids, and true labels if available. For positive examples, the ML label should be 1, and for unlabeled examples, it should be 0. True labels are valid only for the known datasets and are required if you want to compute the classification performance metrics. The “testparams” folder inside the “examples” folder has several “.yaml” files. These files have optimized parameters for XGBoost, CatBoost, and GMM for each test dataset. The value for the XGBoost parameter “n_jobs” is set to 4. You can increase/decrease the value of “n_jobs” depending on the number of CPU cores on your machine. If you want to override the parameters, you can update the file "alpha.yaml" and pass it as an argument. If "alpha.yaml" is not given as an argument, the default parameters of the ML classifiers are used.

There are two steps to use the PULSNAR package:
1. instantiate the *PULSNARClassifier()* with the parameters explained above.

```
# get parameters from user for PULSNAR algorithm.
if len(sys.argv) < 2:
    user_param_file = 'testparams/snar_alpha.yaml'
else:
    user_param_file = sys.argv[1]

# instantiate PULSNARClassifier
pls = PULSNAR.PULSNARClassifier(scar=True, csrdata=False, classifier='xgboost',
                                bin_method='scott', bw_method='hist', lowerbw=0.01, upperbw=0.5, optim='local',
                                calibration=True, calibration_data='PU', calibration_method='isotonic',
                                calibration_n_bins=100, smooth_isotonic=False,
                                classification_metrics=True,
                                n_iterations=1, kfold=5, kflips=1,
                                pulsnar_params_file=user_param_file)
```
2. call the *pulsnar()* function with features, ML labels, record ids, and true labels, if available.

```
res = pls.pulsnar(X, Y, tru_label=Y_true, rec_list=rec_ids)
```
The package returns a dictionary "res" containing estimated alpha, prediction and estimation filenames, and classification performance metrics if true labels were provided.

E.g.
```
{'estimated_alpha': 0.2021, 'prediction_file': 'results/predictions.tsv', 'alpha_file': 'results/alpha_estimates.tsv', 'pulsnar_brier_score': 0.09333678005578955, 'pulsnar_average_precision_score': 0.8715526453230149, 'pulsnar_auc_roc_score': 0.94711006925, 'pulsnar_f1_score': 0.7716913808251601, 'pulsnar_mcc_score': 0.697232752378723, 'pulsnar_accuracy': 0.8787333333333334, 'true_alpha': 0.2}
```
