# -*- coding: utf-8 -*-
# Main file

import argparse
import sys
import tensorflow as tf

FLAGS = None

def makeOutputDir(outputDir, outputName):
    import os
    overwrite= False
    tempOutputDir= outputDir
    tempOutputName= outputName
    while (os.path.exists(tempOutputDir) & (not overwrite)):
        overwrite_string= raw_input(tempOutputDir+" allready exists, overwrite y/n? ")
        if  overwrite_string== "y":
            overwrite= True
        elif overwrite_string=="n":
            fileExtension= raw_input("then enter a file extension! ")
            tempOutputDir= outputDir[:-1]+fileExtension+"/"
            tempOutputName= outputName+fileExtension
    if not os.path.exists(tempOutputDir):
            os.makedirs(tempOutputDir)
    return tempOutputDir, tempOutputName

def main(_):
    """ Runs the gridSearch """
    from clf import someClf
    from evaluation import evaluator
    from gridsearch import multimodelsearch
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    outputDir , outputName = makeOutputDir("../output/", "model")

    someClf = someClf.clfHandler(outputDir, outputName, batch_size=100, training_size=1000)

    parameter = {"batch_size": [100,200,300],
                 "training_size": [1000,2000,3000]
                 }

    scoring = {"Accuracy": "accuracy"}

    clfEvaluator = evaluator.evalHandler(inputDir=outputDir, inputName=outputName)

    parameter = (someClf, parameter, clfEvaluator)

    searcher = multimodelsearch.MultiModelSearch(mnist, parameter, scoring=scoring)

    searcher.fit_and_eval()

    searcher.plotResults(["batch_size", "training_size"], ["Accuracy"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
