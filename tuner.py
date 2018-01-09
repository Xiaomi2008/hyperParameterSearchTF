""" Contains the tuner class for multi model grid search. """
import tensorflow as tf
from tensorflow.contrib.learn import learn_runner

class tuner():
    """ Tuner class """

    def __init__(self,default_param, hparams, FLAGS):
        self.default_param = default_param
        self.hparams_dict = hparams.values()
        self.model_dir = FLAGS.model_dir
        self.old_para = None

        # Set the run_config and the directory to save the model and stats
        run_config = tf.contrib.learn.RunConfig()
        run_config = run_config.replace(model_dir=FLAGS.model_dir)
        self.run_config = run_config

    def next_trial(self):
        """ Get the next experiment """
        for key in self.hparams_dict.keys():
            if self.hparams_dict[key] == []:
                continue
            if type(self.hparams_dict[key]) == list:
                self.default_param.parse(self.old_para)
                self.old_para = (key, self.default_param.key)
                self.default_param.parse((key, self.hparams_dict[key][0]))
                del self.hparams_dict[key][0]
                return True
            if self.hparams_dict[key] != None:
                self.default_param.parse(self.old_para)
                self.old_para = (key, self.default_param.key)
                self.default_param.parse((key, self.hparams_dict[key]))
                return True
        return False

    def run_experiment(self, experiment_fn):
        """Run the training experiment."""
        # Define model parameters
        params = tf.contrib.training.HParams(
            num_hidden_units=100,
            learning_rate=0.002,
            n_classes=10,
            train_steps=5000,
            min_eval_frequency=100
        )

        run_config = tf.contrib.learn.RunConfig()
        run_config = run_config.replace(model_dir=self.model_dir)

        learn_runner.run(
            experiment_fn=experiment_fn,  # First-class function
            run_config=self.run_config,  # RunConfig
            schedule="train_and_evaluate",  # What to run
            hparams=self.default_param  # HParams
        )
