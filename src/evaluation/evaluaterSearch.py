class evaluaterSearch():
    """ This starts the evaluation for a ranker model """

    def __init__(self, evaluater, new_para):
        """ Init of the evaluaterSearch """
        self.evaluater = evaluater
        self.new_para = new_para
        self.results = None

    def evaluate(self, data, process_number, scoring):
        """ Starts to evaluate all saved models and collects the scoring data """
        setattr(self.evaluater, self.new_para[0], self.new_para[1])
        newInputFile = self.evaluater.inputDir + self.evaluater.inputName + "_" + str(process_number)
        setattr(self.evaluater, "inputFile", newInputFile)
        self.evaluater.evaluate(data)
        tmp_result = dict()
        for score in scoring:
            check_score = getattr(self.evaluater, scoring[score])
            if callable(check_score):
                tmp_result[score] = check_score()
        self.results = tmp_result
