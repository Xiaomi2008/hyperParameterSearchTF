""" This class contains the specific evalHandler """

import tensorflow as tf

class evalHandler():
    """ clf evaluator handler """

    def __init__(self, inputDir, inputName):
        self.inputDir = inputDir
        self.inputName = inputName
        self.inputFile = self.inputDir + self.inputName
        self.accuracy_value = None

    def accuracy(self):
        """ AUC Score """
        return self.accuracy_value

    def more_scoring(self):
        """ Add your additional scoring here """

    def evaluate(self, data):
        """ Function how evaluates the saved model """
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            saver = tf.train.import_meta_graph(self.inputFile + "_model.meta", clear_devices=True)
            saver.restore(sess, self.inputFile + "_model")

            x = graph.get_tensor_by_name("x:0")
            y = graph.get_tensor_by_name("y:0")
            y_ = graph.get_tensor_by_name("y_:0")
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.accuracy_value = sess.run(accuracy, feed_dict={x: data.test.images,
                                                y_: data.test.labels})
            print(self.accuracy_value)
