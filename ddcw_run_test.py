from model import DiversifiedDynamicClassWeightedClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.data.data_stream import DataStream
from utils.data_preprocesing import read_kdd_data_multilable, read_data_arff, read_data_csv

# 1.a Load and preprocessing data
"""
Data source: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
"""
#data, X, y = read_kdd_data_multilable('./data/kddcup.data_10_percent_corrected.csv')

# 1.b Load and preprocessing data
"""
Data source: https://github.com/alipsgh/data_streams
"""
#data, X, y = read_data_arff('./data/stagger_w_50_n_0.1_103.arff')
#data, X, y = read_data_arff('./data/led_w_500_n_0.1_104.arff')

# 1.c Load and preprocessing data
"""
Data source: https://github.com/scikit-multiflow/streaming-datasets
"""
#data, X, y = read_data_csv('./data/streaming-datasets-master/elec.csv')
#data, X, y = read_data_csv('./data/streaming-datasets-master/airlines.csv')
#data, X, y = read_data_csv('./data/streaming-datasets-master/agr_a.csv')
#data, X, y = read_data_csv('./data/streaming-datasets-master/covtype.csv')


stream = DataStream(X, y)

stream.prepare_for_use()

# 2a. Models initialization
nb = NaiveBayes()
ht = HoeffdingTreeClassifier()
aw = AccuracyWeightedEnsembleClassifier()
dw = DynamicWeightedMajorityClassifier()
ob = OnlineBoostingClassifier()
oz = OzaBaggingClassifier()

# 2b. Inicialization of DDCW model for comparsion tests
dwc = DiversifiedDynamicClassWeightedClassifier(period=100, base_estimators=[NaiveBayes(), HoeffdingTreeClassifier()], min_estimators=5, max_estimators=20, alpha=0.2, beta=3, theta=0.2) #0.5

# 2c. Inicialization of DDCW models for parameter testing
#ht1_p1 = DiversifiedDynamicClassWeightedClassifier(period=500)
#ht1_p5 = DiversifiedDynamicClassWeightedClassifier(period=500)
#ht1_p10 = DiversifiedDynamicClassWeightedClassifier(period=1000)

#ht1_p1_div = DiversifiedDynamicClassWeightedClassifier(period=100, enable_diversity=True) # default True
#ht1_p5_div = DiversifiedDynamicClassWeightedClassifier(period=500, enable_diversity=True) # default True
#ht1_p10_div = DiversifiedDynamicClassWeightedClassifier(period=1000, enable_diversity=True) # default True

#ht1_p1_disable_div = DiversifiedDynamicClassWeightedClassifier(period=100, enable_diversity=False)
#ht1_p5_disable_div = DiversifiedDynamicClassWeightedClassifier(period=500, enable_diversity=False)
#ht1_p10_disable_div = DiversifiedDynamicClassWeightedClassifier(period=1000, enable_diversity=False)


# 3. Evalution settings
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=1000,
                                batch_size=1,
                                max_samples=1000000,
                                n_wait=1000,
                                metrics=['accuracy', 'f1', 'precision', 'recall', 'running_time', 'model_size'])
                                #metrics=['accuracy'])




# 4. Example of evaluation DDCW compared with other models

evaluator.evaluate(stream=stream, model=[nb, ht, dwc], model_names=['NaiveBayes', 'HoeffdingTree', 'DiversifiedDynamicClassWeighted'])

#evaluator.evaluate(stream=stream, model=[aw, dw, dwc], model_names=['AccurancyWeightedEnsemble', 'DynamicWeightedMajority','DiversifiedDynamicClassWeighted'])

#evaluator.evaluate(stream=stream, model=[ob, oz, dwc], model_names=['OnlineBoosting', 'OzaBagging','DiversifiedDynamicClassWeighted'])

#evaluator.evaluate(stream=stream, model=[dwc, aw, dw, ob, oz, nb], model_names=['DDWC', 'AWE','DWM', 'OB', 'OZB', 'NB'])


# 5. Examples for comparing progress on DDCW with different model parameters
#evaluator.evaluate(stream=stream, model=[ht1_p1, ht1_p5, ht1_p10], model_names=['DDCW-100', 'DDCW-500','DDCW-1000'])

#evaluator.evaluate(stream=stream, model=[ht1_p1_div, ht1_p10_div], model_names=['DDCW-100', 'DDCW-1000'])


import pickle
pickle.dump(evaluator, open("evaluatorobject", "wb"))

