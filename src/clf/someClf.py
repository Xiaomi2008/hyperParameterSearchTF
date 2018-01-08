""" Specific higgsRanker """

import basicClf
import tensorflow as tf

class clfHandler():
    """ This class build a clf model and fits this model to train data """

    def __init__(self, outputDir, outputName, batch_size, training_size):
        """ Init of the clfHandler"""
        self.batch_size = batch_size
        self.training_size = training_size
        self.outputDir = outputDir
        self.outputName = outputName
        self.outputFile = self.outputDir + self.outputName

    def fit(self, data):
        self._build_model()
        self._model.fit(data, self.outputFile)

    def _build_model(self):
        """ Define the structure of the neural net """
        x = tf.placeholder(tf.float32, [None, 784], name="x")
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W, name="y") + b

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        self._model = basicClf.basicClf(x, W, b, y, y_, cross_entropy, train_step,
                                        self.batch_size, self.training_size)

    def returnTrainingScores(self):
        """ Return the mean and the std of the fitting from the basicRanker """
