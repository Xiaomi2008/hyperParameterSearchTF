class clfSearch():
    """Class for hyper parameter search. This class starts the fitting for a clf model """

    def __init__(self, clf, new_para):
        """ clfsearch """
        self.clf = clf
        self.new_para = new_para
        self.trainingScore = None

    def fit(self, data, process_number):
        """ Fit """
        setattr(self.clf, self.new_para[0], self.new_para[1])
        newOutputFile = self.clf.outputDir + self.clf.outputName + "_" + str(process_number)
        setattr(self.clf, "outputFile", newOutputFile)
        self.clf.fit(data)
        self.trainingScore = self.clf.returnTrainingScores()

    def returnTrainingScores(self):
        """ Return the AUC of the training of the basicRanker """
        return self.trainingScore
