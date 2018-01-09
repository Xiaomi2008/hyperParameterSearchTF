# hyperParameterSearchTF
Small setup for hyperParameterSearch with Tensorflow.

# How the program works:

Configure your search in the main.py. There you load your ```classifier.classifier``` and your
 ```evaluator.evaluator```. With some default values. In the example code it is:

```Python
someClf = classifier.classifier(outputDir, outputName,
                batch_size=100, training_size=1000)
```

and

```Python
clfEvaluator = evaluator.evaluator(inputDir=outputDir, inputName=outputName)
```

Then you need to specify your ```parameter = {"batch_size": [100,200,300], 
"training_size": [1000,2000,3000]}``` and ```scoring = {"Accuracy": "accuracy"}```. Then add your
clf and evaluater to ```parameter = (someClf, parameter, clfEvaluator)``` and pass everything to
```searcher = multimodelsearch.MultiModelSearch(mnist, parameter, scoring=scoring)```.

This class will call the classes classifierSearch and evaluaterSearch and they will generate for every
 parameter in your parameter dict classifier.classifier and evaluator.evaluator classes
and will replace only the default values if they are listed in the parameter dict.

The function ```searcher.fit_and_eval()``` is looping over all to parameters and evaluators and returns
the results.

`TODO` Use the multiprocessing library and  ```searcher.fit_and_eval()``` for training and evaluating
parallel. At the moment the lines are not working.

```Python
pool = Pool()
fit_and_eval_results = pool.map(_parallel_fit_eval_number, clf_numbers)
--> cPickle.PicklingError: Can't pickle <type 'module'>: attribute lookup __builtin__.module failed
```
`TODO` Convert the code that it can be used as a tuner object inside the learn_runner.py

```
Tune an experiment with hyper-parameters.
  It iterates trials by running the Experiment for each trial with the
  corresponding hyper-parameters. For each trial, it retrieves the
  hyper-parameters from `tuner`, creates an Experiment by calling experiment_fn,
  and then reports the measure back to `tuner`.
  Example:
```
```Python
    def _create_my_experiment(run_config, hparams):
      hidden_units = [hparams.unit_per_layer] * hparams.num_hidden_layers
      return tf.contrib.learn.Experiment(
          estimator=DNNClassifier(config=run_config, hidden_units=hidden_units),
          train_input_fn=my_train_input,
          eval_input_fn=my_eval_input)
    tuner = create_tuner(study_configuration, objective_key)
    learn_runner.tune(experiment_fn=_create_my_experiment, tuner)
```
```Python
def tune(experiment_fn, tuner):
  while tuner.next_trial():
     tuner.run_experiment(
        _wrapped_experiment_fn_with_uid_check(
        experiment_fn, require_hparams=True))
```
```
 Args:
    experiment_fn: A function that creates an `Experiment`. It should accept an
      argument `run_config` which should be used to create the `Estimator` (
      passed as `config` to its constructor), and an argument `hparams`, which
      should be used for hyper-parameters tuning. It must return an
      `Experiment`.
    tuner: A `Tuner` instance.
```

At the end this results will be plotted by using the file 
(http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py).

# Add your own classifier and evaluator

1. For the classifier you need to add your classifier to the class ```classifier.classifier```. This class should
contain the functions ```fit```, ```_build_model``` and ```returnTrainingScores```. For the hyperparameters
you can add as much as needed. But be careful they need to added into the ```def __init__``` and the default
values need to be set in the main.py as well.

2. For the evaluator you need to add your evaluator to the class ```evaluator.evaluator```. This class should
contain the functions ```evaluate``` and a function for all scoring you want to evaluate. In the example
 only the ```accuracy``` is used. But a sample function called ```more_scoring``` is given.

