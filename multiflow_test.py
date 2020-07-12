from model import DiversifiedDynamicClassWeightedClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OzaBaggingClassifier

from skmultiflow.data.data_stream import DataStream

from utils.data_preprocesing import read_elec_norm_data, read_kdd_data_multilable, read_syntetic_data

# 1. Load and preprocessing data
data, X, y = read_elec_norm_data("./data/elecNormNew.csv")
#data, X, y = read_kdd_data_multilable('./data/kddcup.data_10_percent_corrected.csv')
#data, X, y = read_syntetic_data('./data/stagger_w_50_n_0.1_103.arff')
#data, X, y = read_syntetic_data('./data/led_w_500_n_0.1_104.arff')

stream = DataStream(X, y)

stream.prepare_for_use()

# 2a. Models initialization
nb = NaiveBayes()
ht = HoeffdingTreeClassifier()
aw = AccuracyWeightedEnsembleClassifier()
dw = DynamicWeightedMajorityClassifier()
ob = OnlineBoostingClassifier()
oz = OzaBaggingClassifier()

# 2b. Inicialization of DDCW model
dwc1 = DiversifiedDynamicClassWeightedClassifier()

# 2c. Inicialization of DDCW model for
ht1_p1 = DiversifiedDynamicClassWeightedClassifier(period=100)
ht1_p5 = DiversifiedDynamicClassWeightedClassifier(period=500)
ht1_p10 = DiversifiedDynamicClassWeightedClassifier(period=1000)

ht1_p2 = DiversifiedDynamicClassWeightedClassifier(period=200)
ht1_p3 = DiversifiedDynamicClassWeightedClassifier(period=300)
ht1_p4 = DiversifiedDynamicClassWeightedClassifier(period=400)

ht1_p1_disable_div = DiversifiedDynamicClassWeightedClassifier(period=100, enable_diversity=False)
ht1_p5_disable_div = DiversifiedDynamicClassWeightedClassifier(period=500, enable_diversity=False)
ht1_p10_disable_div = DiversifiedDynamicClassWeightedClassifier(period=1000, enable_diversity=False)


# 3. Evalution settings
evaluator = EvaluatePrequential(show_plot=True,
                                pretrain_size=200,
                                batch_size=1,
                                max_samples=10000,
                                n_wait=200,
                                metrics=['accuracy', 'f1', 'precision', 'running_time', 'model_size'])




#evaluator.evaluate(stream=stream, model=[nb, ht, dwc1], model_names=['NaiveBayes', 'HoeffdingTree', 'DiversifiedDynamicClassWeighted'])

evaluator.evaluate(stream=stream, model=[aw, dw, dwc1], model_names=['AccurancyWeightedEnsemble', 'DynamicWeightedMajority','DiversifiedDynamicClassWeighted'])

#evaluator.evaluate(stream=stream, model=[ob,oz,dwc1], model_names=['OnlineBoosting', 'OzaBagging','DiversifiedDynamicClassWeighted'])

#evaluator.evaluate(stream=stream, model=[ht1_p1, ht1_p5, ht1_p10], model_names=['DDCW-100', 'DDCW-500','DDCW-1000'])

#evaluator.evaluate(stream=stream, model=[ht1_p2, ht1_p3, ht1_p4], model_names=['DDCW-2', 'DDCW-3','DDCW-4'])

#evaluator.evaluate(stream=stream, model=[ht1_p1, ht1_p1_disable_div], model_names=['DDCW-1E', 'DDCW-1D'])





#%%

# Plotting internal model metrics (for DDCW model)
#from utils import plot_custom_model_metrics as pc
#pc.plot_model_size(evaluator, model_ids=[0,1], model_names=['DDCW-1E', 'DDCW-1D'])
#pc.plot_diversity(evaluator, model_ids=[0,1], model_names=['DDCW-1E', 'DDCW-1D'])

