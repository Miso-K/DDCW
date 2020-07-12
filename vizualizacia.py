# library
import numpy as np
import matplotlib.pyplot as plt
from utils.data_preprocesing import read_elec_norm_data, read_kdd_data_multilable, read_syntetic_data

import warnings
warnings.filterwarnings("ignore")

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) # values cannot be negative
    array += 0.0000001 # values cannot be 0
    array = np.sort(array) # values must be sorted
    index = np.arange(1,array.shape[0]+1) # index per array element
    n = array.shape[0] # number of array elements
    return np.sum((2 * index - n  - 1) * array) / (n * np.sum(array)) # Gini coefficient


# data, X, y = read_elec_norm_data("./data/elecNormNew.csv")
# data, X, y = read_kdd_data_multilable('./data/kddcup.data_10_percent_corrected.csv')
# data = data[:100000]
# data, X, y = read_syntetic_data('./data/stagger_w_50_n_0.1_103.arff')
data, X, y = read_syntetic_data('./data/led_w_500_n_0.1_104.arff')


offset = 1000
end = len(data)
gn = []
y = []
x = []

for i, j in enumerate(range(0, end, offset)):
    swdata = np.swapaxes(data[j:j+offset, :-1], 0, 1)
    for k, s in enumerate(swdata):
        gn.append(gini(s)*100)
        y.append(k)
        x.append(j)


plt.figure(figsize=(16,6))
plt.suptitle('LED Gini index')
plt.scatter(x, y, s=gn, marker='o', c=y)
plt.xlabel('Samples')
plt.ylabel('Class')

plt.yticks(np.arange(0, 24, 1))
plt.show()



