import pickle

"""Run visualization of internal metrics of DDCW model after evaluation (read saved evaluation object)"""
evaluator = pickle.load(open("evaluatorobject", "rb"))
# Plotting internal model metrics (for DDCW model)
from utils import plot_custom_model_metrics as pc
pc.plot_model_size(evaluator, model_ids=[0,1], model_names=['DDCW-100', 'DDCW-1000'], plot_title='Progress of experts in DDCW with and without diversity')
pc.plot_diversity(evaluator, model_ids=[0,1], model_names=['DDCW-E', 'DDCW-D'], plot_title='Progress of diversity in DDCW with and without diversity')
pc.plt.show()
