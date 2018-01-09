# hyperParameterSearchTF
Small setup for hyperParameterSearch with Tensorflow.

# How the program works:

Configure your search in the main.py. There you load your someClf.clfHandler and your
 evaluator.evaluatorHandler. With some default values. In the example code it is:

```Python
someClf = someClf.clfHandler(outputDir, outputName, 
                batch_size=100, training_size=1000)
```

and

```Python
clfEvaluator = evaluator.evalHandler(inputDir=outputDir, inputName=outputName)
```

Then you need to specify your ```parameter = {"batch_size": [100,200,300], 
"training_size": [1000,2000,3000]}``` and ```scoring = {"Accuracy": "accuracy"}```. Then add your
clf and evaluater to ```parameter = (someClf, parameter, clfEvaluator)``` and pass everything to
```searcher = multimodelsearch.MultiModelSearch(mnist, parameter, scoring=scoring)```.

This class will generate for every parameter in your parameter dict clfHandler and evalHandler classes
and will replace only the default values if they are listed in the parameter dict.

The function ```searcher.fit_and_eval()``` is using the multiprocessing library and will train and
 evaluate all generated models and after this it will return a dict with the results.

At the end this results will be plotted by using the file 
(http://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py).
