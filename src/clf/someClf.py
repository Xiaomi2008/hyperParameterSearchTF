""" Specific higgsRanker """

import basicClf

class clfHandler():
    """ This class build a clf model and fits this model to train data """

    def __init__(self, outputDir, outputName, layers, feature_func=difference,
                       weight_func=summed_weight, cost_func=weighted_least_square,
                       optimizer=tf.train.AdamOptimizer(0.01), activation=tf.nn.tanh,
                       use_bias=False, kernel_initializer=tf.random_normal_initializer(),
                       steps=20, max_iters=2000, n_samples=1000, max_samples=3000):
        """ Init of the clfHandler"""

    def fit(self, x, y, z=None):
        self._build_model()
        self._model.fit(x, y, z)

    def _build_model(self):
        """ Define the structure of the neural net """
        self._model = basicClf.basicClf()

    def returnTrainingScores(self):
        """ Return the mean and the std of the fitting from the basicRanker """
