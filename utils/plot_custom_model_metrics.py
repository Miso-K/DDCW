import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def get_gradual_metric(metric):

    list_metric = []
    sum_perc = 0
    for i in range(len(metric)):
        sum_perc += metric[i]
        if i == 0:
            list_metric.append(metric[i])
        else:
            avg = sum_perc / (i + 1)
            list_metric.append(avg)
    return list_metric


def plot_model_size(evaluator, model_ids, model_names):

    x_range = []
    n_experts = []
    n_experts_mean = []

    for model_id in model_ids:
        n_experts_tmp = []
        x_range = []
        for measure in evaluator.model[model_id].custom_measurements:
            x_range.append((measure['id_period'] - 1)*100)
            n_experts_tmp.append(measure['n_experts'])
        n_experts.append(n_experts_tmp)
        n_experts_mean.append(get_gradual_metric(n_experts_tmp))

    plt.figure(figsize=(10,4))

    # Pocet expertov v modeli

    avg1 = plt.subplot()
    for id,i in enumerate(model_ids):
        print(id,i, x_range, n_experts[i])
        avg1.plot(x_range, n_experts[i], 'C'+str(i), label='Model ' + model_names[i] + ' (200 samples)')
        avg1.plot(x_range, n_experts_mean[i], ':C'+str(i+2), label='Model ' + model_names[i] + ' (mean)')
    avg1.legend()
    avg1.set_xlabel('Sampless')
    avg1.set_ylabel('n experts')
    avg1.set_title('Progress experts in models', fontsize=8)




def plot_diversity(evaluator, model_ids, model_names):

    x_range = []
    diversities = []
    diversities_mean = []

    for model_id in model_ids:
        diversities_tmp = []
        x_range = []
        for measure in evaluator.model[model_id].custom_measurements:
            x_range.append((measure['id_period'] - 1)*100)
            diversities_tmp.append(measure['diversity'] if not np.isnan(measure['diversity']) else 1)
        diversities.append(diversities_tmp)
        diversities_mean.append(get_gradual_metric(diversities_tmp))


    plt.figure(figsize=(10,4))
    # Diverzita

    avg1 = plt.subplot()
    for id,i in enumerate(model_ids):
        avg1.plot(x_range, diversities[i], 'C'+str(i), label='Model ' + model_names[i] + ' (200 samples)')
        avg1.plot(x_range, diversities_mean[i], ':C'+str(i+2), label='Model ' + model_names[i] + ' (mean)')
    avg1.legend()
    avg1.set_xlabel('Sampless')
    avg1.set_ylabel('Q stat diversity')
    avg1.set_title('Progress diversity in models', fontsize=8)