""" Contains the class for multi model grid search. """
from functools import partial
from matplotlib import pyplot as plt
from multiprocessing import Pool
import numpy as np
from evaluation import evaluaterSearch
from clf import clfSearch


def _parallel_fit_eval(process_number, x_train, y_train, z_train, x_test, y_test, z_test, clf, evaluators, scoring):
    """ Function for fitting and evaluating a model. It returns the results after fitting and evaluation. """
    clf[process_number].fit(x_train, y_train, z_train, process_number)

    results = dict()

    results["train_score_" + str(process_number)] = (clf[process_number].returnTrainingScores())

    evaluators[process_number].evaluate(x_test, y_test, z_test, process_number, scoring)

    for key in evaluators[process_number].results.keys():
        results[key + '_test_score_' + str(process_number)] = [evaluators[process_number].results[key]]

    return results


class MultiModelSearch(object):
    """
    Grid-Search over multiple rankers and parameter tuples.

    Parameters:
    -----------
    data:           data object

    parameter:      tuple or list of tuples
                    Has the form of (estimator, parameters) or a list of those
                    tuples.
    scoring:        string, callable, list/tuple, dict or None, default: None
                    Metrics used for evaluating the search results

    Return:
    -------
    {
    'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                             mask = [False False False False]...)
    'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                            mask = [ True  True False False]...),
    'param_degree': masked_array(data = [2.0 3.0 -- --],
                             mask = [False False  True  True]...),
    'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
    'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
    'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
    'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
    'rank_test_score'    : [2, 4, 3, 1],
    'split0_train_score' : [0.8, 0.9, 0.7],
    'split1_train_score' : [0.82, 0.5, 0.7],
    'mean_train_score'   : [0.81, 0.7, 0.7],
    'std_train_score'    : [0.03, 0.03, 0.04],
    'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
    'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
    'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
    'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
    'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
    }
    """

    def __init__(self, data, parameter, scoring=None):
        self.data = data
        self.parameter = parameter
        self.scoring = scoring
        self.results = dict()
        self._generate_clfs()
        self._generate_evaluaters()

    def _generate_clfs(self):
        """ Function for generating all clf which will be trained """
        clfs = []
        for para_key in self.parameter[1]:
            for value in self.parameter[1][para_key]:
                clfs.append(clfSearch.clfSearch(self.parameter[0], [para_key, value]))
                self._generate_results(self.parameter[0], para_key, value)
        self.clfs = clfs

    def _generate_evaluaters(self):
        """ Function for generating all evaluators for scoring the trained clfs """
        evaluators = []
        for para_key in self.parameter[1]:
            for value in self.parameter[1][para_key]:
                evaluators.append(evaluaterSearch.evaluaterSearch(self.parameter[2], [para_key, value]))
        self.evaluators = evaluators

    def _update_result(self, results, clf_numbers):
        """ Updates the scores in the result dict after an evaluator finished """
        # ToDo make results of scoring values dynamic
        names_results = ['AUC_train_score', 'AUC_test_score', 'train_time']
        # 'mean_fit_time' 'std_fit_time' 'AUC_train_score'
        for number in clf_numbers:
            for name in names_results:
                if name not in self.results:
                    self.results[name] = [results[number][name + "_" + str(number)]]
                else:
                    self.results[name].append(results[number][name + "_" + str(number)])

    def _generate_results(self, ranker, para_key, value):
        """ Puts the choosen values for each ranker to the result dict """
        for att in dir(ranker):
            if not att.startswith('_'):
                if att == 'X' or att == 'Y' or att == 'W':
                    continue
                else:
                    if 'param_' + att in self.results.keys():
                        if att == para_key:
                            self.results['param_' + att].append(value)
                        else:
                            self.results['param_' + att].append(getattr(ranker, att))
                    else:
                        if att == para_key:
                            self.results['param_' + att] = [value]
                        else:
                            self.results['param_' + att] = [getattr(ranker, att)]

    def fit_and_eval(self):
        """ Running the training for all rankers and all parameters """
        ranker_numbers = range(len(self.clfs))
        x_train, y_train, z_train = self.data.get_train()
        x_test, y_test, z_test = self.data.get_test()
        _parallel_fit_eval_number = partial(_parallel_fit_eval, x_train=x_train, y_train=y_train, z_train=z_train,
                                            x_test=x_test, y_test=y_test, z_test=z_test, rankers=self.clfs,
                                            evaluators=self.evaluators, scoring=self.scoring)
        pool = Pool()
        fit_and_eval_results = pool.map(_parallel_fit_eval_number, ranker_numbers)
        self._update_result(fit_and_eval_results, ranker_numbers)

    def plotResults(self, resultName, scoringName):
        """ From http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html """
        plt.figure(figsize=(13, 13))
        plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
                  fontsize=16)

        plt.xlabel(resultName[0])
        plt.ylabel("Score")
        plt.grid()

        ax = plt.axes()
        ax.set_xlim(0, len(self.results['param_' + resultName[0]]))
        ax.set_ylim(0, 1)

        # Get the regular numpy array from the MaskedArray
        X_axis = np.array(range(len(self.results['param_' + resultName[0]])), dtype=float)
        xticks = []
        for xpoint in self.results['param_' + resultName[0]]:
            xticks.append(str(xpoint))
        plt.xticks(X_axis, xticks)
        for scorer, color in zip(sorted(scoringName), ['g', 'k']):
            for sample, style in (('train', '--'), ('test', '-')):
                sample_score_mean = self.results['%s_%s_score' % (scorer, sample)]
                sample_score_std = self.results['std_%s_%s' % (sample, scorer)]
                ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                                sample_score_mean + sample_score_std,
                                alpha=0.1 if sample == 'test' else 0, color=color)
                ax.plot(X_axis, sample_score_mean, style, color=color,
                        alpha=1 if sample == 'test' else 0.7,
                        label="%s (%s)" % (scorer, sample))
            best_index = np.where(self.results['%s_test_score' % scorer] ==
                                  np.array(self.results['%s_test_score' % (scorer)]).max())[0][0]
            best_score = self.results['%s_test_score' % scorer][best_index]
            # If there are more best_scores then take the first one
            if type(best_score) == list:
                best_score = best_score[0]
            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                    linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score,
                        (X_axis[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid('off')
        plt.show()
