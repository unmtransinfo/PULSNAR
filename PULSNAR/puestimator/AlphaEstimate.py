import logging
import numpy as np
import traceback
import scipy.stats as stats
from scipy.optimize import minimize_scalar, dual_annealing, differential_evolution
from sklearn.neighbors import KernelDensity
from scipy.spatial import distance
from scipy.stats import gaussian_kde
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

numpy2ri.activate()
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


def IQR_val(probs):
    """
    This function computes interquartile range of the data i.e. Q3-Q1. The IQR is used by Freedman–Diaconis
    rule to find the number of bins in the given data.

    Parameters
    ----------
    probs: ML predicted probabilities

    Returns
    -------
    the difference between Q3 and Q1
    """

    Q3, Q1 = np.percentile(probs, [75, 25])
    return Q3 - Q1


def nearest_power_two(v):
    """
    This function computes the nearest power of 2 for a given value.

    Parameters
    ----------
    v: a given value

    Returns
    -------
    nearest power of 2.
    """

    lo = 2 ** np.floor(np.log2(v))
    up = 2 ** np.ceil(np.log2(v))
    if v - lo < up - v:
        return lo
    else:
        return up


def compute_bin_count(prbs, bin_method="scott"):
    """
    Compute the optimal bin count for the data using the given method and return nearest 2's power.
    The default method is scott's method.

    Parameters
    ----------
    prbs: ML predicted probabilities
    bin_method: binning method to find the number of methods

    Returns
    -------
    number of bins
    """
    nbins = None
    lp = len(prbs)
    if bin_method == "square_root":
        nbins = nearest_power_two(np.sqrt(lp))
    elif bin_method == "sturges":
        nbins = np.ceil(np.log2(lp)) + 1
    elif bin_method == "rice":
        nbins = nearest_power_two(2 * lp ** (1 / 3))
    elif bin_method == "scott":
        h = 3.5 * np.std(prbs) / lp ** (1 / 3)
        nbins = nearest_power_two((max(prbs) - min(prbs)) / h)
    elif bin_method == "fd":
        h = 2 * IQR_val(prbs) / lp ** (1 / 3)
        nbins = nearest_power_two((max(prbs) - min(prbs)) / h)
    else:
        traceback.print_stack()
        logging.error("Invalid bin computation method provided")
        exit(-1)
    return nbins


def compute_hist_pdf(prbs, n_bins):
    """
    This function generates probability density for the given data using histogram function.

    Parameters
    ----------
    prbs: ML predicted probabilities
    n_bins: number of bins

    Returns
    -------
    normalized probability density
    """

    bins = np.linspace(0, 1, n_bins + 1)
    hdensity, _ = np.histogram(prbs, bins=bins, density=True)
    return hdensity / sum(hdensity)


def compute_gaussian_kde(prbs, bw, n_bins):
    """
    This function generates probability density for the given data using gaussian kde function.

    Parameters
    ----------
    prbs: ML predicted probabilities
    n_bins: number of bins
    bw: bandwidth

    Returns
    -------
    normalized probability density
    """

    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(prbs.reshape(-1, 1))
    x = np.linspace(0, 1, n_bins)[:, np.newaxis]
    kdensity = np.exp(kde.score_samples(x))
    return kdensity / sum(kdensity)


def beta_kernel(v_bins, data=None, lmbda=None):
    """
    This function computes beta kernels by Chen (1999) CSDA.

    Parameters
    ----------
    v_bins: bin list
    data: ML predicted probabilities
    lmbda: bandwidth

    Returns
    -------
    beta kernel density for a given bandwidth
    """

    if data is None or lmbda is None:
        traceback.print_stack()
        logging.error("Data/BW are missing")
        exit(-1)
    else:
        return np.asarray([np.mean(stats.beta.pdf(data, bn / lmbda + 1, (1 - bn) / lmbda + 1)) for bn in v_bins])


def compute_beta_kernel_density(prbs, n_bins=512, bw=0.01):
    """
    Compute beta kernel density for positive and unlabeled probabilities

    Parameters
    ----------
    prbs: ML predicted probabilities
    n_bins: number of bins
    bw: bandwidth

    Returns
    -------
    normalized beta kernel density
    """

    x = np.linspace(0, 1, n_bins)
    bpdf = beta_kernel(x, data=prbs, lmbda=bw)
    bpdf = bpdf / sum(bpdf)
    return x, bpdf


def minimize_mse(b, probs, n_bins, hdensity):
    """
    Function for mean square error using histogram density and beta/gaussian kernel density.

    Parameters
    ----------
    b: bandwidth
    probs: ML predicted probabilities
    n_bins: number of bins
    hdensity: pdf using histogram

    Returns
    -------
    distance between beta kernel density and histogram density
    """

    _, kdensity = compute_beta_kernel_density(probs, n_bins=n_bins, bw=b)
    # kdensity = compute_gaussian_kde(probs, b, n_bins)
    return np.mean(pow(kdensity - hdensity, 2))
    # return distance.jensenshannon(kdensity, hdensity)


def err_function(estrange, unlab_kde=None, case_kde=None):
    """
    This function computes error for each point in estrange

    Parameters
    ----------
    estrange: range of the estimated alpha
    unlab_kde: kernel density for unlabeled tests
    case_kde: kernel density for cases

    Returns
    -------
    list of minimum error at each point in estrange
    """

    if unlab_kde is None or case_kde is None:
        traceback.print_stack()
        logging.error("unlabeled/case density is not provided")
        exit(-1)
    else:
        return np.asarray([(unlab_kde - case_kde * z).min() for z in estrange])


class PositiveFractionEstimation:
    def __init__(self, bin_method="scott", lowerbw=0.01, upperbw=0.5, bw_method="hist", optim="global"):
        self.bin_method = bin_method
        self.lowerbw = lowerbw  # smallest BW
        self.upperbw = upperbw  # largest BW
        self.bw_method = bw_method
        self.optim = optim
        self.estrange = [i / 10000 for i in range(10000)]  # estimation range

    def estimate_positive_fraction_in_unlabeled(self, probs, ml_label):
        """
        Compute the proportion of positive tests in the unlabeled set using probability density method.

        Parameters
        ----------
        probs: predicted probabilities
        ml_label: ML labels

        Returns
        -------
        estimated alpha (fraction of positives in the unlabeled)
        """

        idx0 = np.where(ml_label == 0)[0]
        idx1 = np.where(ml_label == 1)[0]
        probs0 = probs[idx0]
        probs1 = probs[idx1]

        # compute number of bins for determining BW
        n_bins = min(512, int(compute_bin_count(probs, self.bin_method)))

        # compute bandwidth using the given method
        bw = None
        if self.bw_method == "ucv" or self.bw_method == "bcv" or self.bw_method == "nrd" or self.bw_method == "nrd0":
            bw = self.calculate_bandwidth_r(probs)
        elif self.bw_method == "hist":
            bw = self.calculate_bandwidth_hist(probs, n_bins)
        else:
            traceback.print_stack()
            logging.error("wrong bandwidth method")
            exit(-1)

        # estimate fraction of positives in the unlabeled
        x, casey = compute_beta_kernel_density(probs1, n_bins=n_bins, bw=bw)
        _, unlaby = compute_beta_kernel_density(probs0, n_bins=n_bins, bw=bw)
        epsilon = 1e-10 if abs(min(casey)) == 0 else abs(min(casey))
        unlabyk = unlaby[x > 0.5]
        caseyk = casey[x > 0.5]

        # compute error for each value in estrange and then use slope of error to find alpha
        err = err_function(self.estrange, unlab_kde=unlabyk, case_kde=caseyk)
        estrange1 = self.estrange[1:]
        d = np.diff(np.log10(abs(err) + epsilon)) / np.diff(self.estrange)
        # print(d, max(d), np.argmax(d))
        i = np.where(abs(np.diff(d)) == max(abs(np.diff(d))))[0][0]
        return estrange1[i]

    def calculate_bandwidth_r(self, data):
        """
        This function calls R methods to calculate the bandwidth

        Parameters
        ----------
        data: list of probabilities

        Returns
        -------
        bandwidth
        """
        bw = None
        utils = importr('stats')
        if self.bw_method == "ucv":
            bw = utils.bw_ucv(data)[0]
        elif self.bw_method == "bcv":
            bw = utils.bw_bcv(data)[0]
        elif self.bw_method == "nrd":
            bw = utils.bw_nrd(data)[0]
        elif self.bw_method == "nrd0":
            bw = utils.bw_nrd0(data)[0]
        else:
            traceback.print_stack()
            logging.error("wrong R method to calculate bandwidth")
            exit(-1)
        # numpy2ri.deactivate()
        # print("bw_r: ", bw)
        return bw

    def calculate_bandwidth_hist(self, data, bin_count):
        """
        This function uses histogram method to calculate the bandwidth.

        Parameters
        ----------
        data: list of probabilities
        bin_count: number of bins

        Returns
        -------
        bandwidth
        """

        # compute pdf using histogram
        hist_dens = compute_hist_pdf(data, bin_count)
        bw = None
        if self.optim == "local":
            ret = minimize_scalar(minimize_mse, args=(data, bin_count, hist_dens), bounds=(self.lowerbw, self.upperbw),
                                  method='Bounded')
            bw = ret.x
        elif self.optim == "global":
            lw = [self.lowerbw]
            up = [self.upperbw]
            # ret = dual_annealing(minimize_mse, bounds=list(zip(lw, up)), args=(data, bin_count, hist_dens),
            #                      no_local_search=True, maxiter=1000, seed=7)
            ret = differential_evolution(minimize_mse, bounds=list(zip(lw, up)), args=(data, bin_count, hist_dens))
            bw = ret.x[0]
        else:
            traceback.print_stack()
            logging.error("wrong optimization method. it should be global or local")
            exit(-1)
        # print("bw_hist: ", bw)
        return bw
