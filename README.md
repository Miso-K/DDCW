# Diversified Dynamic Class Weighted Classifier (DDCW)

This repository contains implementation of new ensemble model (DDCW) and scripts for testing model

[![Python version](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/downloads/release/python-350/)

### Install requirements
```
pip install -i requirements.txt
```

### Manual instalation
```
git clone https://github.com/Miso-K/DDCW
```

Requirements:
```
pip install -U numpy
pip install -U Cython
pip install -U arff
pip install -U scikit-multiflow
```

### Run DDCW model tests
```
python3 ddcw_run_test.py
```

### Run other scripts
Run custom metrics of DDCW model (works only on DDCW model):
```
python3 custom_model_metrics.py
```

Run Gini index visualization of dataset per window (eg. 100 samples)
```
python3 gini_index_visualize.py
```