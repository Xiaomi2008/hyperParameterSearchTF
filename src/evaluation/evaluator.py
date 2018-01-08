""" This class contains the specific evalHandler """

class evalHandler():
    """ clf evaluator handler """

    def __init__(self, inputDir, inputName, feature_func, weight_func):
        self.inputDir = inputDir
        self.inputName = inputName
        self.inputFile = self.inputDir + self.inputName
        self.feature_func = feature_func
        self.weight_func = weight_func
        self.pc = None
        self.auc = None

    def pc(self):
        """ PC Score """
        return self.pc

    def roc_auc(self):
        """ AUC Score """
        return self.auc

    def more_scoring(self):
        """ Add your additional scoring here """

    def evaluate(self, x, y, z):
        """
        Function how evaluates the saved model

        """
