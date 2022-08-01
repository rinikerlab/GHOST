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
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import metrics
from scipy import stats
import os


def calc_classification_metric(y_true, y_pred, ThOpt_metrics='Kappa'):
    """ Calculate the selected classification metric

    Parameters
    ----------
    y_true: 1d array-like, or label indicator array / sparse matrix
        Ground truth labels
    y_pred: 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier

    Returns
    ----------
    score: float
        Return the selected classification score
    """
    ThOpt_metrics=ThOpt_metrics.lower()
    if not ThOpt_metrics in ['kappa', 'cohen_kappa_score', 'balanced_accuracy', 'balanced_accuracy_score', 
                         'accuracy', 'accuracy_score', 'f1', 'f1_score', 'mcc', 'matthews_corrcoef', 
                         'precision', 'precision_score', 'recall', 'recall_score', 'sensitivity', 'specificity',
                         ]:
        print("WARNING: The specified optmization metric could not be found. "
              "WARNING: The Cohen's kappa will be used instead.")
        ThOpt_metrics='kappa'
    try:
        if ThOpt_metrics == 'kappa' or ThOpt_metrics == 'cohen_kappa_score':
            tgt=metrics.cohen_kappa_score(y_true,y_pred)
        elif ThOpt_metrics == 'balanced_accuracy' or ThOpt_metrics == 'balanced_accuracy_score':
            tgt=metrics.balanced_accuracy_score(y_true,y_pred)
        elif ThOpt_metrics == 'accuracy' or ThOpt_metrics == 'accuracy_score':
            tgt=metrics.accuracy_score(y_true, y_pred)
        elif ThOpt_metrics == 'f1' or ThOpt_metrics == 'f1_score':
            tgt=metrics.f1_score(y_true,y_pred)
        elif ThOpt_metrics == 'mcc' or ThOpt_metrics == 'matthews_corrcoef':
            tgt=metrics.matthews_corrcoef(y_true,y_pred)
        elif ThOpt_metrics == 'precision' or ThOpt_metrics == 'precision_score':
            tgt=metrics.precision_score(y_true,y_pred)
        elif ThOpt_metrics == 'recall' or ThOpt_metrics == 'recall_score' or ThOpt_metrics == 'sensitivity':
            tgt=metrics.recall_score(y_true,y_pred)
        elif ThOpt_metrics == 'specificity':
            tn, fp, fn, tp=metrics.confusion_matrix(y_true, y_pred,
                                                      labels=sorted(set(y_true))).ravel()
            tgt=tn/(tn+fp)
        return tgt
    except:
        print("ERROR: The optimization metric could not be calculated")
        return np.nan


def gen_subsets(df_preds, labels, subsets_size, random_seed, with_replacement=False):
    """ Function used by GHOST threshold optimization to randomly subsample the training set

    Parameters
    ----------
    df_preds: pandas dataframe
        dataframe containing true labels, prediction probabilities and predictions at the input thresholds
        for the training set
    labels: sequence of ints
        Ground truth labels for the training set
    subsets_size: float or int
        Size of the subsets. if float, represents the proportion of the dataset to include in the subsets.
        If integer, it represents the actual number of instances to include in the subsets.
    with_replacement: bool
        The subsets are drawn randomly. True to draw the subsets with replacement
    random_seed: int
        random number to seed the drawing of the subset
    """
    if with_replacement:
        if isinstance(subsets_size, float):
            Nsamples=int(df_preds.shape[0]*subsets_size)
        elif isinstance(subsets_size, int):
            Nsamples=subsets_size
        df_subset=resample(df_preds, replace=True, n_samples=Nsamples, stratify=labels,
                             random_state=random_seed)
        labels_subset=df_subset['labels']
    else:
        df_tmp, df_subset, labels_tmp, labels_subset=train_test_split(df_preds, labels,
                                                                        test_size=subsets_size,
                                                                        stratify=labels,
                                                                        random_state=random_seed)
    return df_subset, labels_subset

def clear_plots(plt):
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    return

def plot_opt_metric_vs_thresholds(thresholds, opt_metr_accum=None, median_opt_metr=None,
                                  ThOpt_metrics='Kappa', average='threshold', greater_is_better=True,
                                  figure_folder='.', figure_basename='test'):
    import matplotlib.pyplot as plt
    if plt:
        clear_plots(plt)
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    """Plot the threshold optimization curve"""
    # set colors
    N_subsets=len(thresholds)
    np.random.seed(0)
    colors=[]
    for i in range(N_subsets):
        colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))
    if opt_metr_accum is not None:
        if greater_is_better:
            th_accum=[thresholds[np.argmax(opt_metr_accum[i])] for i in range(N_subsets)]
        else:
            th_accum=[thresholds[np.argmin(opt_metr_accum[i])] for i in range(N_subsets)]
        for i in range(N_subsets):
            plt.scatter(thresholds, opt_metr_accum[i], alpha=0.5, c=colors[i])
            plt.plot(thresholds, opt_metr_accum[i], alpha=0.5, c=colors[i])
            if average == 'threshold':
                plt.axvline(th_accum[i], ls='--', lw=1, c=colors[i], alpha=0.7)
        if average == 'threshold':
            plt.axvline(np.median(th_accum), ls='--', c='red', lw=3)
    if average == 'curve' and median_opt_metr is not None:
        plt.plot(thresholds, median_opt_metr, c='red', alpha=0.7, lw=3)
        plt.scatter(thresholds, median_opt_metr, c='red', marker='x')
        if greater_is_better:
            plt.axvline(thresholds[np.nanargmax(median_opt_metr)], ls='--', c='red', lw=3)
        else:
            plt.axvline(thresholds[np.nanargmin(median_opt_metr)], ls='--', c='red', lw=3)

    plt.xlabel('decision threshold', size=18)
    plt.ylabel(ThOpt_metrics, size=18)
    plt.ylim(0, 1)
    plt.savefig(f'{figure_folder}/plt_threshold_opt_{figure_basename}.png', dpi=300, bbox_inches='tight')
    return plt



def plot_opt_metric_vs_hp_shifts(hp_shift_accum, candidate_shifts_accum, opt_metr_accum, 
                                 bin_edges=None, median_shift=None, median_opt_metr=None,
                                 ThOpt_metrics='Kappa', average='threshold', greater_is_better=True,
                                 figure_folder='.', figure_basename='test'):
    import matplotlib.pyplot as plt
    if plt:
        clear_plots(plt)
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    #set colors
    np.random.seed(0)
    colors=[]
    for i in range(len(opt_metr_accum)):
        colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))

    argsort_candidate_shifts_accum=[ np.argsort(row1) for row1 in candidate_shifts_accum ]
    candidate_shifts_accum_sort=[ np.array(row1)[argsort_candidate_shifts_accum[i]] for i, row1 in enumerate(candidate_shifts_accum) ]
    opt_metr_accum_sort=[ np.array(row1)[argsort_candidate_shifts_accum[i]] for i, row1 in enumerate(opt_metr_accum) ]
    for i in range(len(opt_metr_accum_sort)):
        plt.scatter(candidate_shifts_accum_sort[i], opt_metr_accum_sort[i], alpha=0.5, c=colors[i])
        plt.plot(candidate_shifts_accum_sort[i], opt_metr_accum_sort[i], alpha=0.5, c=colors[i])
        if average=='threshold':
            plt.axvline(hp_shift_accum[i], ls='--', lw=1, c=colors[i], alpha=0.7)
    if average=='threshold':    
        plt.axvline(np.median(hp_shift_accum), ls='--', c='red', lw=3)
    elif average=='curve':
        if bin_edges is not None:
            for i in bin_edges:
                plt.axvline(i, ls='--', c='gray')
        if median_shift is not None and median_opt_metr is not None:
            plt.plot(median_shift, median_opt_metr, c='red', alpha=0.7, lw=3)
            plt.scatter(median_shift, median_opt_metr, c='red', marker='x')
            if greater_is_better:
                plt.axvline(median_shift[np.nanargmax(median_opt_metr)], ls='--', c='red', lw=3)
            else:
                plt.axvline(median_shift[np.nanargmin(median_opt_metr)], ls='--', c='red', lw=3)

    plt.xlabel('hyperplane shift', size=18)
    plt.ylabel(ThOpt_metrics, size=18)
    plt.ylim(0, 1)
    plt.savefig(f'{figure_folder}/plt_hyperplane_opt_{figure_basename}.png', dpi=300, bbox_inches='tight')
    return plt


def optimize_threshold_from_predictions(labels, probs, thresholds=np.round(np.arange(0.05,1.0,0.05),2),
                                        ThOpt_metrics='Kappa', greater_is_better=True,
                                        N_subsets=100, subsets_size=0.2, with_replacement=False,
                                        random_seed=None, average='curve',
                                        plot_optimization_curve=False, figure_folder='.', figure_basename='test'):

    """ Optimize the decision threshold based on subsets of the training set.
    The threshold that maximizes the Cohen's kappa coefficient or a ROC-based criterion 
    on the training subsets is chosen as optimal.
    
    Parameters
    ----------
    labels: sequence of ints
        True labels for the training set
    probs: sequence of floats
        prediction probabilities of the class with the “greater label” (cls.classes_[1])
        for the training samples (e.g. output from cls.predict_proba(X_train)[:,1])
    thresholds: list of floats
        List of decision thresholds to screen for classification
    ThOpt_metrics: str
        Optimization metric ('kappa', 'mcc', 'f1', 'accuracy', 'balanced_accuracy',
        'precision', 'recall', 'specificity', 'ROC'). Default is 'kappa'
    greater_is_better: bool
        if True (default), GHOSTML will return the threshold that maximizes the optimization metric.
        Otherwise, GHOSTML will return the threshold that minimizes the optimization metric.
    N_subsets: int
        Number of training subsets to use in the optimization
    subsets_size: float or int
        Size of the subsets. if float, represents the proportion of the dataset to include in the subsets. 
        If integer, it represents the actual number of instances to include in the subsets. 
    with_replacement: bool
        The subsets are drawn randomly. True to draw the subsets with replacement
    random_seed: int    
        random number to seed the drawing of the subsets
    average: str
        if 'curve' (default), it calculates the median classification metric per threshold over the N_subsets
            and returns the threshold that optimizes the median optimization curve
        if 'threshold', it calculates the optimal threshold for every subset
            and returns the median optimal threshold over the N_subsets
    plot_optimization_curve: bool
        if True, a figure displaying the optimization curve (optimization metric vs. candidate thresholds) will be generated as output
    figure_folder: str
        folder path where to store the output figure if plot_optimization_curve is True. Default is "."
    figure_basename: str
        basename for the output figure returned if plot_optimization_curve is True. Default is "test"

    Returns
    ----------
    thresh: float
        Optimal decision threshold for classification
    figure: png file
        If plot_optimization_curve is True, the output plot of the outimization curve is stored in
        figure_folder/plt_threshold_opt_{figure_basename}.png
    """
    ThOpt_metrics = ThOpt_metrics.lower()
    # seeding
    np.random.seed(random_seed)
    random_seeds=np.random.randint(N_subsets*10, size=N_subsets)  
    
    df_preds=pd.DataFrame({'labels':labels,'probs':probs})
    thresh_names=[str(x) for x in thresholds]
    for thresh in thresholds:
        df_preds[str(thresh)]=[1 if x>=thresh else 0 for x in probs]
    # Optmize the decision threshold based on the Cohen's Kappa coefficient
    opt_metrics_accum=[]
    if ThOpt_metrics != 'roc':
        # pick N_subsets training subsets and determine the threshold that provides the highest kappa on each 
        # of the subsets
        for i in range(N_subsets):
            df_subset, labels_subset=gen_subsets(df_preds, labels, subsets_size, random_seeds[i],
                                                   with_replacement=with_replacement)
            opt_metric_subset=[]
            for col1 in thresh_names:
                opt_metric_subset.append(calc_classification_metric(labels_subset, list(df_subset[col1]),
                                                                  ThOpt_metrics=ThOpt_metrics))

            opt_metrics_accum.append(opt_metric_subset)
        # determine the threshold that provides the best results on the training subsets
        if average == 'threshold':
            y_values_median=None
            if greater_is_better:
                opt_thresh=np.median([thresholds[np.argmax(opt_metrics_accum[i])] for i in range(N_subsets)])
            else:
                opt_thresh=np.median([thresholds[np.argmin(opt_metrics_accum[i])] for i in range(N_subsets)])
        elif average == 'curve':
            y_values_median, y_values_std=helper_calc_median_std(opt_metrics_accum)
            if greater_is_better:
                opt_thresh=thresholds[np.argmax(y_values_median)]
            else:
                opt_thresh=thresholds[np.argmin(y_values_median)]
        if plot_optimization_curve:
            plot_opt_metric_vs_thresholds(thresholds, opt_metr_accum=opt_metrics_accum, median_opt_metr=y_values_median,
                                          ThOpt_metrics=ThOpt_metrics, average=average,
                                          greater_is_better=greater_is_better,
                                          figure_folder=figure_folder, figure_basename=figure_basename)
    # Optmize the decision threshold based on the ROC-curve, as described here https://doi.org/10.1007/s11548-013-0913-8
    elif ThOpt_metrics == 'roc':
        sensitivity_accum=[]
        specificity_accum=[]
        # Calculate sensitivity and specificity for a range of thresholds and N_subsets
        for i in range(N_subsets):
            df_subset, labels_subset=gen_subsets(df_preds, labels, subsets_size, random_seeds[i],
                                                   with_replacement=with_replacement)
            sensitivity=[]
            specificity=[]
            for thresh in thresholds:
                scores=[1 if x >= thresh else 0 for x in df_subset['probs']]
                tn, fp, fn, tp=metrics.confusion_matrix(labels_subset, scores, labels=sorted(set(labels))).ravel()
                sensitivity.append(tp/(tp+fn))
                specificity.append(tn/(tn+fp))
            sensitivity_accum.append(sensitivity)
            specificity_accum.append(specificity)
            if plot_optimization_curve or average=='threshold':
                opt_metrics_accum.append((2*np.array(sensitivity)*np.array(specificity))/(np.array(sensitivity)+np.array(specificity)))
        # determine the threshold that provides the best results on the training subsets
        if average == 'curve':
            median_sensitivity, std_sensitivity=helper_calc_median_std(sensitivity_accum)
            median_specificity, std_specificity=helper_calc_median_std(specificity_accum)
            roc_dist_01corner=(2*median_sensitivity*median_specificity)/(median_sensitivity+median_specificity)
            if greater_is_better:
                opt_thresh=thresholds[np.argmax(roc_dist_01corner)]
            else:
                opt_thresh=thresholds[np.argmin(roc_dist_01corner)]
        elif average=='threshold':
            roc_dist_01corner=None
            if greater_is_better:
                opt_thresh=np.median([thresholds[np.argmax(opt_metrics_accum[i])] for i in range(N_subsets)])
            else:
                opt_thresh=np.median([thresholds[np.argmin(opt_metrics_accum[i])] for i in range(N_subsets)])
        if plot_optimization_curve:
            plot_opt_metric_vs_thresholds(thresholds, opt_metr_accum=opt_metrics_accum, median_opt_metr=roc_dist_01corner,
                                          ThOpt_metrics=ThOpt_metrics, average=average,
                                          greater_is_better=greater_is_better,
                                          figure_folder=figure_folder, figure_basename=figure_basename)
    return opt_thresh


def optimize_threshold_from_oob_predictions(labels_train, oob_probs, thresholds, ThOpt_metrics='Kappa',
                                            greater_is_better=True,
                                            plot_optimization_curve=False, figure_folder='.', figure_basename='test'
                                            ):
    """Optimize the decision threshold based on the prediction probabilities of the out-of-bag set of random forest.
    The threshold that maximizes the Cohen's kappa coefficient or a ROC-based criterion 
    on the out-of-bag set is chosen as optimal.
    
    Parameters
    ----------
    labels_train: list of int
        True labels for the training set
    oob_probs : list of floats
        prediction probabilities of the class with the “greater label” (cls.classes_[1])
        for the out-of-bag set of a trained random forest model
    thresholds: list of floats
        List of decision thresholds to screen for classification
    ThOpt_metrics: str
        Optimization metric ('kappa', 'mcc', 'f1', 'accuracy', 'balanced_accuracy',
        'precision', 'recall', 'specificity', 'ROC'). Default is 'kappa'
    greater_is_better: bool
        if True (default), GHOSTML will return the threshold that maximizes the optimization metric.
        Otherwise, GHOSTML will return the threshold that minimizes the optimization metric.
    plot_optimization_curve: bool
        if True, a figure displaying the optimization curve (optimization metric vs. candidate thresholds) will be generated as output
    figure_folder: str
        folder path where to store the output figure if plot_optimization_curve is True. Default is "."
    figure_basename: str
        basename for the output figure returned if plot_optimization_curve is True. Default is "test"

    Returns
    ----------
    thresh: float
        Optimal decision threshold for classification
    figure: png file
        If plot_optimization_curve is True, the output plot of the outimization curve is stored in
        figure_folder/plt_threshold_opt_{figure_basename}.png
    """
    ThOpt_metrics = ThOpt_metrics.lower()
    # Optmize the decision threshold based on the Cohen's Kappa coefficient
    if ThOpt_metrics != 'roc':
        tscores=[]
        opt_metric_accum = []
        # evaluate the score on the oob using different thresholds
        for thresh in thresholds:
            scores=[1 if x>=thresh else 0 for x in oob_probs]
            opt_metric=calc_classification_metric(labels_train, scores, ThOpt_metrics=ThOpt_metrics)
            tscores.append((np.round(opt_metric,3),thresh))
        # select the threshold providing the highest/lowest clasiification score as optimal
        if plot_optimization_curve:
            plot_opt_metric_vs_thresholds(thresholds, opt_metr_accum=None, median_opt_metr=[x[0] for x in tscores],
                                          ThOpt_metrics=ThOpt_metrics, average='curve',
                                          greater_is_better=greater_is_better,
                                          figure_folder=figure_folder, figure_basename=figure_basename)
        if greater_is_better:
            tscores.sort(reverse=True)
        else:
            tscores.sort(reverse=False)
        thresh=tscores[0][-1]

    # Optmize the decision threshold based on the ROC-curve
    elif ThOpt_metrics == 'roc':
        # ROC optimization with thresholds determined by the roc_curve function of sklearn
        fpr, tpr, thresholds_roc=metrics.roc_curve(labels_train, oob_probs, pos_label=1)
        specificity=1-fpr
        roc_dist_01corner=(2*tpr*specificity)/(tpr+specificity)
        if greater_is_better:
            thresh=thresholds_roc[np.argmax(roc_dist_01corner)]
        else:
            thresh=thresholds_roc[np.argmin(roc_dist_01corner)]
        if plot_optimization_curve:
            plot_opt_metric_vs_thresholds(thresholds_roc, opt_metr_accum=None, median_opt_metr=roc_dist_01corner,
                                          ThOpt_metrics=ThOpt_metrics, average='curve',
                                          greater_is_better=greater_is_better,
                                          figure_folder=figure_folder, figure_basename=figure_basename)
    return thresh


def helper_calc_median_std(specificity):
    # Calculate median and std of the columns of a pandas dataframe
    arr=np.array(specificity)
    y_values_median=np.median(arr,axis=0)
    y_values_std=np.std(arr,axis=0)
    return y_values_median, y_values_std


def svm_othr_from_predictions(labels, probs, ThOpt_metrics='Kappa', greater_is_better=True,
                              N_subsets=100, subsets_size=0.2, with_replacement=False,
                              random_seed=None, average='threshold',
                              plot_optimization_curve=False, figure_folder='.', figure_basename='test'):
    """ Optimize the hyperplane position on subsets of the training set.
    This combines the GHOST workflow with the SVM-OTHR approach
    proposed by Yu et al. 2015 (https://doi.org/10.1016/j.knosys.2014.12.007).
    The hyperplane shift that optimizes the chosen classification metric is chosen as optimal.

    Parameters
    ----------
    labels: sequence of ints
        True labels for the training set
    probs: sequence of floats
        predict confidence scores for the training samples (e.g. output from cls.decision_function(X_train)).
        The confidence score for a sample has to be proportional to the signed distance of that sample to the hyperplane.
    ThOpt_metrics: str
        Optimization metric ('kappa', 'mcc', 'f1', 'accuracy', 'balanced_accuracy',
        'precision', 'recall', 'specificity', 'ROC'). Default is 'kappa'
    greater_is_better: bool
        if True (default), GHOST will return the hyperplane shift that maximizes the optimization metric.
        Otherwise, GHOST will return the hyperplane shift that minimizes the optimization metric.
    N_subsets: int
        Number of training subsets to use in the optimization
    subsets_size: float or int
        Size of the subsets. if float, represents the proportion of the dataset to include in the subsets.
        If integer, it represents the actual number of instances to include in the subsets.
    with_replacement: bool
        The subsets are drawn randomly. True to draw the subsets with replacement
    random_seed: int
        random number to seed the drawing of the subsets
    average: str
        if 'curve' (default), it calculates the median classification metric per hyperplane shift over the N_subsets
            and returns the hyperplane shift that optimizes the median optimization curve
        if 'threshold', it calculates the hyperplane shift for every subset
            and returns the median optimal hyperplane shift over the N_subsets
    plot_optimization_curve: bool
        if True, a figure displaying the optimization curve (optimization metric vs. candidate hyperplane shifts)
        will be generated as output
    figure_folder: str
        folder path where to store the output figure if plot_optimization_curve is True. Default is "."
    figure_basename: str
        basename for the output figure returned if plot_optimization_curve is True. Default is "test"

    Returns
    ----------
    hyperplane_shift: float
        Optimal hyperplane shift for SVM or RIDGE classifiers
    figure: png file
        If plot_optimization_curve is True, the output plot of the outimization curve is stored in
        figure_folder/plt_hyperplane_opt_{figure_basename}.png
    """
    ThOpt_metrics = ThOpt_metrics.lower()
    np.random.seed(random_seed)
    random_seeds=np.random.randint(N_subsets * 10, size=N_subsets)
    df_preds=pd.DataFrame({'labels': labels, 'probs': probs})

    hp_shift_accum=[]
    candidate_shifts_accum=[]
    opt_metr_accum=[]
    if ThOpt_metrics == 'roc':
        sensitivity_accum=[]
        specificity_accum=[]
        # Calculate sensitivity and specificity for a range of thresholds and N_subsets
        for i in range(N_subsets):
            df_subset, labels_subset=gen_subsets(df_preds, labels, subsets_size, random_seeds[i],
                                                   with_replacement=with_replacement)
            hp_shift, candidate_shifts1, sensitivity=svm_othr_from_distances(labels_subset,
                                                                               list(df_subset['probs']),
                                                                               ThOpt_metrics='sensitivity',
                                                                               greater_is_better=greater_is_better,
                                                                               return_more=True)
            hp_shift, candidate_shifts2, specificity=svm_othr_from_distances(labels_subset,
                                                                               list(df_subset['probs']),
                                                                               ThOpt_metrics='specificity',
                                                                               greater_is_better=greater_is_better,
                                                                               return_more=True)
            if candidate_shifts1 == candidate_shifts2:
                candidate_shifts_accum.append(candidate_shifts1)
            else:
                print('error')
            sensitivity_accum.append(sensitivity)
            specificity_accum.append(specificity)
            roc_dist_01corner_tmp=(2 * np.array(sensitivity) * np.array(specificity)) / (
                        np.array(sensitivity) + np.array(specificity))
            opt_metr_accum.append(roc_dist_01corner_tmp)
            hp_shift_accum.append(candidate_shifts1[np.nanargmax(roc_dist_01corner_tmp)])

            # bin over candidate shifts
        if average == 'threshold':
            if plot_optimization_curve:
                plot_opt_metric_vs_hp_shifts(hp_shift_accum, candidate_shifts_accum, opt_metr_accum,
                                             ThOpt_metrics=ThOpt_metrics, average='threshold',
                                             greater_is_better=greater_is_better,
                                             figure_folder=figure_folder, figure_basename=figure_basename)
            return np.median(hp_shift_accum)

        elif average == 'curve':
            sensitivity_flat=[item for sublist in sensitivity_accum for item in sublist]
            specificity_flat=[item for sublist in specificity_accum for item in sublist]
            candidate_shifts_flat=[item for sublist in candidate_shifts_accum for item in sublist]
            median_sensitivity, bin_edges, binnumber=stats.binned_statistic(candidate_shifts_flat, sensitivity_flat,
                                                                              statistic='median', bins=20)
            median_specificity, bin_edges, binnumber=stats.binned_statistic(candidate_shifts_flat, specificity_flat,
                                                                              statistic='median', bins=20)
            median_shift, bin_edges, binnumber=stats.binned_statistic(candidate_shifts_flat, candidate_shifts_flat,
                                                                        statistic='median', bins=20)

            roc_dist_01corner=(2 * median_sensitivity * median_specificity) / (
                        median_sensitivity + median_specificity)
            if plot_optimization_curve:
                plot_opt_metric_vs_hp_shifts(hp_shift_accum, candidate_shifts_accum, opt_metr_accum,
                                             bin_edges=bin_edges, median_shift=median_shift,
                                             median_opt_metr=roc_dist_01corner,
                                             ThOpt_metrics=ThOpt_metrics, average='curve',
                                             greater_is_better=greater_is_better,
                                             figure_folder=figure_folder, figure_basename=figure_basename)
            return median_shift[np.nanargmax(roc_dist_01corner)]

    else:
        # pick N_subsets training subsets and determine the threshold that provides the highest kappa on each
        # of the subsets
        for i in range(N_subsets):
            df_subset, labels_subset=gen_subsets(df_preds, labels, subsets_size, random_seeds[i],
                                                   with_replacement=with_replacement)
            # svm-othr optimization
            # if shift == 'hyperplane': else 'threshold' ## for integrated ghostml function
            hp_shift, candidate_shifts, opt_metr=svm_othr_from_distances(labels_subset,
                                                                           list(df_subset['probs']),
                                                                           ThOpt_metrics=ThOpt_metrics,
                                                                           greater_is_better=greater_is_better,
                                                                           return_more=True)
            hp_shift_accum.append(hp_shift)
            if plot_optimization_curve or average == 'curve':
                candidate_shifts_accum.append(candidate_shifts)
                opt_metr_accum.append(opt_metr)

        if average == 'threshold':
            if plot_optimization_curve:
                plot_opt_metric_vs_hp_shifts(hp_shift_accum, candidate_shifts_accum, opt_metr_accum,
                                             ThOpt_metrics=ThOpt_metrics, average='threshold',
                                             greater_is_better=greater_is_better,
                                             figure_folder=figure_folder, figure_basename=figure_basename)
            return np.median(hp_shift_accum)

        elif average == 'curve':
            opt_metr_flat=[item for sublist in opt_metr_accum for item in sublist]
            candidate_shifts_flat=[item for sublist in candidate_shifts_accum for item in sublist]
            median_opt_metr, bin_edges, binnumber=stats.binned_statistic(candidate_shifts_flat, opt_metr_flat,
                                                                           statistic='median', bins=20)
            median_shift, bin_edges, binnumber=stats.binned_statistic(candidate_shifts_flat, candidate_shifts_flat,
                                                                        statistic='median', bins=20)
            if plot_optimization_curve:
                plot_opt_metric_vs_hp_shifts(hp_shift_accum, candidate_shifts_accum, opt_metr_accum,
                                             bin_edges=bin_edges, median_shift=median_shift,
                                             median_opt_metr=median_opt_metr,
                                             ThOpt_metrics=ThOpt_metrics, average='curve',
                                             greater_is_better=greater_is_better,
                                             figure_folder=figure_folder, figure_basename=figure_basename)

            return median_shift[np.nanargmax(median_opt_metr)]


def svm_othr_from_distances(y_true, cls_decision_function,
                            ThOpt_metrics='Kappa', greater_is_better=True, return_more=False):
    """SVM-OTHR approach proposed by Yu et al. 2015 (https://doi.org/10.1016/j.knosys.2014.12.007)"""
    y_pred=[1 if x >= 0 else 0 for x in cls_decision_function]
    df_svm=pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'dist': cls_decision_function})
    majority_class=np.argmax(np.unique(y_true, return_counts=True)[1])
    minority_class=np.argmin(np.unique(y_true, return_counts=True)[1])
    df_svm_minority=df_svm.loc[df_svm['y_true'] == minority_class]
    df_svm_majority=df_svm.loc[df_svm['y_true'] == majority_class]
    # take the misclassified instances of the minority class
    df_svm_minority_wrong=df_svm_minority.loc[df_svm_minority['y_true'] != df_svm_minority['y_pred']]
    distances_minority_wrong=list(df_svm_minority_wrong['dist'])
    # for every misclassified minority instance,
    # calculate the distance to the nearest neighbour majority instance far from the hyperplane
    if all(x < 0 for x in distances_minority_wrong):
        negative=True
    elif all(x > 0 for x in distances_minority_wrong):
        negative=False
    else:
        print("Warning: something is wrong")
    dist_nn=[]
    for i, ist1 in enumerate(distances_minority_wrong):
        try:
            if negative:
                dist_nn.append(max(df_svm_majority.loc[df_svm_majority['dist'] <= ist1, 'dist']))
            else:
                dist_nn.append(min(df_svm_majority.loc[df_svm_majority['dist'] >= ist1, 'dist']))
        except:
            dist_nn.append(np.nan)

    candidate_positions = list((-np.array(distances_minority_wrong) -np.array(dist_nn)) / 2)
    candidate_positions=[0] + [x for x in candidate_positions if not np.isnan(x)]

    # calculate the optimization metric with new hyperplane
    opt_metric=[]
    for i, hp1 in enumerate(candidate_positions):
        # df_svm[f'dist_hp{i}'] =
        scores=df_svm['dist'] + candidate_positions[i]
        y_pred_new=[1 if x >= 0 else 0 for x in scores]
        # opt_metric.append(metrics.cohen_kappa_score(y_true, y_pred_new))
        opt_metric.append(calc_classification_metric(y_true, y_pred_new, ThOpt_metrics=ThOpt_metrics))

    if greater_is_better:
        # return hyperplane shift maximizing the optimization metric
        hyperplane_shift=candidate_positions[np.argmax(opt_metric)]
    else:
        # return hyperplane shift minimizing the optimization metric
        hyperplane_shift=candidate_positions[np.argmin(opt_metric)]
    if return_more:
        return hyperplane_shift, candidate_positions, opt_metric
    else:
        return hyperplane_shift


def svm_othr_from_estimator(cls, X, y_true, ThOpt_metrics='Kappa', greater_is_better=True):
    """SVM-OTHR approach proposed by Yu et al. 2015 (https://doi.org/10.1016/j.knosys.2014.12.007)"""
    cls_decision_function=cls.decision_function(X)
    hyperplane_shift=svm_othr_from_distances(y_true, cls_decision_function,
                                               ThOpt_metrics=ThOpt_metrics,
                                               greater_is_better=greater_is_better)
    return hyperplane_shift


