# -*- coding: utf-8 -*-
# Main file


def main():
    import tensorflow as tf
    from evaluation import evaluator
    from gridsearch import multimodelsearch
    from helpers.funcs import difference, weighted_least_square, summed_weight
    from clf import someClf

    """ Runs the gridSearch """

    outputDir = "yourDir"
    outputName = "yourName"
    data = "yourData"

    someClf = someClf.clfHandler(outputDir, outputName, layers=[5, 5, 1], feature_func=difference,
                                         weight_func=summed_weight, cost_func=weighted_least_square,
                                         optimizer=tf.train.AdamOptimizer(0.01), activation=tf.nn.tanh,
                                         use_bias=False, kernel_initializer=tf.random_normal_initializer(),
                                         steps=20, max_iters=2000, n_samples=1000, max_samples=3000)

    parameter = {"layers": [[1, 2, 3], [1, 5, 6], [5, 5, 1], [5, 5, 1], [5, 5, 1], [5, 5, 1], [5, 5, 1]],
                           "steps": [20, 30, 40, 50],
                           "max_iters": [2000, 3000, 4000],
                           "n_samples": [1000, 2000, 3000]
                 }

    scoring = {"AUC": "roc_auc", "PC": "pc"}

    clfEvaluator = evaluator.evalHandler(inputDir=outputDir,
                                         inputName=outputName,
                                         feature_func=difference,
                                         weight_func=summed_weight
                                         )

    parameter = (someClf, parameter, clfEvaluator)

    searcher = multimodelsearch.MultiModelSearch(data, parameter, scoring=scoring)

    searcher.fit_and_eval()

    searcher.plotResults(["layers", "steps", "max_iters", "n_samples"], ["AUC"])


if __name__ == "__main__":
    main()
