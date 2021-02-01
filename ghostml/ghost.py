#
# Copyright (C) 2021 Carmen Esposito and GHOST contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def optimize_threshold_from_predictions(labels, probs, thresholds, 
                                    ThOpt_metrics = 'Kappa', N_subsets = 100, 
                                    subsets_size = 0.2, with_replacement = False, random_seed = None):

    """ Optimize the decision threshold based on subsets of the training set.
    The threshold that maximizes the Cohen's kappa coefficient or a ROC-based criterion 
    on the training subsets is chosen as optimal.
    
    Parameters
    ----------
    labels: sequence of ints
        True labels for the training set
    probs: sequence of floats
        predicted probabilities for minority class from the training set 
        (e.g. output from cls.predict_proba(data)[:,1])
    thresholds: list of floats
        List of decision thresholds to screen for classification
    ThOpt_metrics: str
        Optimization metric. Choose between "Kappa" and "ROC"
    N_subsets: int
        Number of training subsets to use in the optimization
    subsets_size: float or int
        Size of the subsets. if float, represents the proportion of the dataset to include in the subsets. 
        If integer, it represents the actual number of instances to include in the subsets. 
    with_replacement: bool
        The subsets are drawn randomly. True to draw the subsets with replacement
    random_seed: int    
        random number to seed the drawing of the subsets
    
    Returns
    ----------
    thresh: float
        Optimal decision threshold for classification
    """
    # seeding
    np.random.seed(random_seed)
    random_seeds = np.random.randint(N_subsets*10, size=N_subsets)  
    
    df_preds = pd.DataFrame({'labels':labels,'probs':probs})
    thresh_names = [str(x) for x in thresholds]
    for thresh in thresholds:
        df_preds[str(thresh)] = [1 if x>=thresh else 0 for x in probs]
    # Optmize the decision threshold based on the Cohen's Kappa coefficient
    if ThOpt_metrics == 'Kappa':
        # pick N_subsets training subsets and determine the threshold that provides the highest kappa on each 
        # of the subsets
        kappa_accum = []
        for i in range(N_subsets):
            if with_replacement:
                if isinstance(subsets_size, float):
                    Nsamples = int(df_preds.shape[0]*subsets_size)
                elif isinstance(subsets_size, int):
                    Nsamples = subsets_size                    
                df_subset = resample(df_preds, replace=True, n_samples = Nsamples, stratify=labels, random_state = random_seeds[i])
                labels_subset = df_subset['labels']
            else:
                df_tmp, df_subset, labels_tmp, labels_subset = train_test_split(df_preds, labels, test_size = subsets_size, stratify = labels, random_state = random_seeds[i])
            kappa_train_subset = []
            for col1 in thresh_names:
                kappa_train_subset.append(metrics.cohen_kappa_score(labels_subset, list(df_subset[col1])))
            kappa_accum.append(kappa_train_subset)
        # determine the threshold that provides the best results on the training subsets
        y_values_median, y_values_std = helper_calc_median_std(kappa_accum)
        opt_thresh = thresholds[np.argmax(y_values_median)]
    # Optmize the decision threshold based on the ROC-curve, as described here https://doi.org/10.1007/s11548-013-0913-8
    elif ThOpt_metrics == 'ROC':
        sensitivity_accum = []
        specificity_accum = []
        # Calculate sensitivity and specificity for a range of thresholds and N_subsets
        for i in range(N_subsets):
            if with_replacement:
                if isinstance(subsets_size, float):
                    Nsamples = int(df_preds.shape[0]*subsets_size)
                elif isinstance(subsets_size, int):
                    Nsamples = subsets_size                    
                df_subset = resample(df_preds, n_samples = Nsamples, stratify=labels, random_state = random_seeds[i])
                labels_subset = list(df_subset['labels'])
            else:
                df_tmp, df_subset, labels_tmp, labels_subset = train_test_split(df_preds, labels, test_size = subsets_size, stratify = labels, random_state = random_seeds[i])
            sensitivity = []
            specificity = []
            for thresh in thresholds:
                scores = [1 if x >= thresh else 0 for x in df_subset['probs']]
                tn, fp, fn, tp = metrics.confusion_matrix(labels_subset, scores, labels=sorted(set(labels))).ravel()
                sensitivity.append(tp/(tp+fn))
                specificity.append(tn/(tn+fp))
            sensitivity_accum.append(sensitivity)
            specificity_accum.append(specificity)
        # determine the threshold that provides the best results on the training subsets
        median_sensitivity, std_sensitivity = helper_calc_median_std(sensitivity_accum)
        median_specificity, std_specificity = helper_calc_median_std(specificity_accum)
        roc_dist_01corner = (2*median_sensitivity*median_specificity)/(median_sensitivity+median_specificity)
        opt_thresh = thresholds[np.argmax(roc_dist_01corner)]
    return opt_thresh


def helper_calc_median_std(specificity):
    # Calculate median and std of the columns of a pandas dataframe
    arr = np.array(specificity)
    y_values_median = np.median(arr,axis=0)
    y_values_std = np.std(arr,axis=0)
    return y_values_median, y_values_std    

