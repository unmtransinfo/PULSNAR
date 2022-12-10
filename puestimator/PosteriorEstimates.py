import logging
import traceback
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)


class PosteriorEstimation:
    def __init__(self, scar=False):
        self.scar = scar

    def compute_posterior_probs(self, preds_dict, ml_label_dict, true_label_dict, cal_preds_dict=None):
        """
        for non_scar data, compute posterior probability using the predicted probabilities per cluster.

        Parameters
        ----------
        preds_dict: a dictionary with record id as key and predicted probabilities as value
        ml_label_dict: a dictionary with record id as key and ml label as value
        true_label_dict: a dictionary with record id as key and true label as value
        cal_preds_dict: a dictionary with record id as key and calibrated probabilities as value

        Returns
        -------
        probs: list of posterior probabilities
        ml_label: list of ml labels
        true_label: list of true labels
        recs: list of record ids
        cal_probs: list of calibrated posterior probabilities
        """
        probs = []
        ml_label = []
        true_label = []
        recs = []
        cal_probs = []
        if self.scar:
            traceback.print_stack()
            logging.error("this operation is not supported for SCAR data")
            exit(-1)
        else:
            for kk, vv in ml_label_dict.items():
                recs.append(kk)
                ml_label.append(min(vv, key=vv.count))
                true_label.append(np.mean(true_label_dict[kk]).astype(int))
                if len(vv) > 1:  # unlabeled records
                    psx = 1 - np.prod(1 - np.asarray(preds_dict[kk]))
                    if cal_preds_dict is not None:
                        cal_psx = 1 - np.prod(1 - np.asarray(cal_preds_dict[kk]))
                else:  # positive records
                    psx = np.mean(preds_dict[kk])
                    if cal_preds_dict is not None:
                        cal_psx = np.mean(cal_preds_dict[kk])
                probs.append(psx)
                if cal_preds_dict is not None:
                    cal_probs.append(cal_psx)
        return np.asarray(probs), np.asarray(ml_label), np.asarray(true_label), np.asarray(recs), np.asarray(cal_probs)
