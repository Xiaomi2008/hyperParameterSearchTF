""" Basic ranker class with the fit function """

import tensorflow as tf

class basicClf():
    """ Clf class which is used for fitting a built model """

    def __init__(self, x, W, b, y, y_, cross_entropy, train_step, batch_size, training_size):
        """ Init of the basicClf """
        self.x = x
        self.W = W
        self.b = b
        self.y = y
        self.y_ = y_
        self.cross_entropy = cross_entropy
        self.train_step = train_step
        self.batch_size = batch_size
        self.training_size = training_size

    def fit(self, data, outputFile):
        """ Train the model """
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            for _ in range(self.training_size):
                batch_xs, batch_ys = data.train.next_batch(self.batch_size)
                sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
        saver.save(sess, outputFile + "_model")

    def returnTrainingScores(self):
        """ Return the Scores of the training """
