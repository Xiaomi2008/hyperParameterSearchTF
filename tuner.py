""" Contains the tuner class for multi model grid search. """
import tensorflow as tf
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib.training import HParams

class tuner():
    """ Tuner class """
    def __init__(self,default_param, hparams, model_dir):
        self.default_param_dict = default_param.values()
        self.hparams_dict = hparams.values()
        self.old_para = None

        # Set the run_config and the directory to save the model and stats
        run_config = tf.contrib.learn.RunConfig()
        run_config = run_config.replace(model_dir=model_dir)
        self.run_config = run_config

    def next_trial(self):
        """ Get the next experiment """
        for key in self.hparams_dict.keys():
            if self.hparams_dict[key] == []:
                continue
            if type(self.hparams_dict[key]) == list:
                if self.old_para != None:
                    self.default_param_dict[self.old_para[0]] = self.old_para[1]
                self.old_para = (key, self.default_param_dict[key])
                self.default_param_dict[key] = self.hparams_dict[key][0]
                del self.hparams_dict[key][0]
                return True
            if self.hparams_dict[key] != None:
                self.default_param_dict.parse(self.old_para)
                self.old_para = (key, self.default_param_dict[key])
                self.default_param_dict[key] = self.hparams_dict[key]
                return True
        return False

    def run_experiment(self, experiment_fn):
        """Run the training experiment."""
        # Define model parameters
        hparams = HParams()
        for k, v in self.default_param_dict.items():
            hparams.add_hparam(k, v)

        learn_runner.run(
            experiment_fn=experiment_fn,  # First-class function
            run_config=self.run_config,  # RunConfig
            schedule="train_and_evaluate",  # What to run
            hparams=hparams  # HParams
        )
