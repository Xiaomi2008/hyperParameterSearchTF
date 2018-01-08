# -*- coding: utf-8 -*-
# Class how handles the gridsearch
# ToDo:
import inspect

import tensorflow as tf
from evaluation import evaluator
from gridsearch import multimodelsearch
from helpers.funcs import difference, weighted_least_square, summed_weight
from ranking import ranker


def gridSearch(dataPath):
    """ Runs the gridSearch """

    higgsRanker = ranker.higgsRanker(outputDir, outputName, layers=[5, 5, 1], feature_func=difference,
                                     weight_func=summed_weight, cost_func=weighted_least_square,
                                     optimizer=tf.train.AdamOptimizer(0.01), activation=tf.nn.tanh,
                                     use_bias=False, kernel_initializer=tf.random_normal_initializer(),
                                     steps=20, max_iters=2000, n_samples=1000, max_samples=3000)

    parameter_higgs = {"layers": [[1, 2, 3], [1, 5, 6], [5, 5, 1], [5, 5, 1], [5, 5, 1], [5, 5, 1], [5, 5, 1]],
                       "steps": [20, 30, 40, 50],
                       "max_iters": [2000, 3000, 4000],
                       "n_samples": [1000, 2000, 3000]
                       }

    scoring = {"AUC": "roc_auc", "PC": "pc"}

    logging.info('Initialize the higgsEvaluator')
    higgsEvaluator = evaluator.higgsEvaluator(inputDir=outputDir,
                                              inputName=outputName,
                                              feature_func=difference,
                                              weight_func=summed_weight)
    logging.info(inspect.getmembers(higgsEvaluator))

    parameter = (higgsRanker, parameter_higgs, higgsEvaluator)

    logging.info('Initialize the searcher')
    searcher = multimodelsearch.MultiModelSearch(data_handler, parameter, logging, scoring=scoring)
    logging.info(inspect.getmembers(searcher))

    logging.info('Fit the searcher')
    searcher.fit_and_eval()

    logging.info('Evaluate the searcher')
    # searcher.evaluate()

    logging.info('Plot the results of the gridsearch')
    searcher.plotResults(["layers", "steps", "max_iters", "n_samples"], ["AUC"])

"""
    evaluation = evaluation.RankEvaluation()
    ranking = evaluation.evaluate(searcher.results, scoring, no_splits=5,
                                  scoring_weight=[2.0, 1.0, 4.0])

    for a, b in ranking:
        print("Parameter: {}\nScore: {}".format(a, b))
"""
