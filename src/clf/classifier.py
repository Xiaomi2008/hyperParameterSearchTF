""" Specific higgsRanker """

import tensorflow as tf

class classifier():
    """ This class build a clf model and fits this model to train data """

    def __init__(self, outputDir, outputName, batch_size, training_size):
        """ Init of the clfHandler"""
        self.batch_size = batch_size
        self.training_size = training_size
        self.outputDir = outputDir
        self.outputName = outputName
        self.outputFile = self.outputDir + self.outputName
        self._build_model()
        self.trainingScore = None

    def fit(self, data):
        """ Train the model """
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            for _ in range(self.training_size):
                batch_xs, batch_ys = data.train.next_batch(self.batch_size)
                sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
            saver.save(sess, self.outputFile + "_model")

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

        self.x = x
        self.W = W
        self.b = b
        self.y = y
        self.y_ = y_
        self.cross_entropy = cross_entropy
        self.train_step = train_step
        self.trainingScore = None

    def returnTrainingScores(self):
        """ Return the mean and the std of the fitting from the basicRanker """
        return self.trainingScore
