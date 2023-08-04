import os
import numpy as np
import pandas as pd

import time
import sys
import argparse
import h5py

import matplotlib
matplotlib.use('Agg')
import viz



current_dir = os.path.dirname(os.path.abspath(__file__))
deepsurv_dir = os.path.join(current_dir, "deepsurv")


sys.path.insert(1, deepsurv_dir)


import utils
from collections import defaultdict
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import logging
import uuid
import time

localtime   = time.localtime()
TIMESTRING  = time.strftime("%m%d%Y%M", localtime)
DURATION_COL = 'time'
EVENT_COL = 'censor'


data_dir = os.path.join(current_dir, "experiments/data/whas")

data_filepath =os.path.join(data_dir, "whas_train_test.h5")
whas_dirpath = os.path.join(current_dir, "whas")

def h5_to_array(hdf5_group):

    x_array = np.array(hdf5_group['x'])
    t_array = np.array(hdf5_group['t'])
    e_array = np.array(hdf5_group['e'])

    return x_array, t_array, e_array


def evaluate_model(model, dataset, bootstrap = False):
    def ci(model):
        def cph_ci(x, t, e, **kwargs):
            return concordance_index(
                event_times= t, 
                predicted_scores= -model.predict_partial_hazard(x), 
                event_observed= e,
            )
        return cph_ci

    def mse(model):
        def cph_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(-model.predict_partial_hazard(x).values)
            return ((hr_pred - hr) ** 2).mean()
        return cph_mse  

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = ci(model)(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(ci(model), dataset)
    
    # Calcualte MSE
    if 'hr' in dataset:
        metrics['mse'] = mse(model)(**dataset)
        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics

# /trinity/home/hmo/hmo/dl_sa_tutorial/experiments/data/whas/whas_train_test.h5

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='name of the experiment that is being run')
    parser.add_argument('dataset', help='.h5 File containing the train/valid/test datasets')
    parser.add_argument('--results_dir', default='/shared/results', help='Directory to save resulting models and visualizations')
    parser.add_argument('--plot_error', action="store_true", help="If arg present, plot absolute error plots")
    parser.add_argument('--treatment_idx', default=None, type=int, help='(Optional) column index of treatment variable in dataset. If present, run treatment visualizations.')
    # return parser.parse_args()

    args = parser.parse_args()
    print("Arguments:",args)



    # Load Dataset
    print("Loading datasets: " + args.dataset)
    datasets = utils.load_datasets(args.dataset)

    print ("datasets['test']", datasets['test'])

    # Train CPH model
    print("Training CPH Model")
    train_df = utils.format_dataset_to_df(datasets['train'], DURATION_COL, EVENT_COL)
    
    print (train_df)

    cf = CoxPHFitter()
    results = cf.fit(train_df, duration_col=DURATION_COL, event_col=EVENT_COL)
    cf.print_summary()
    print("Train Likelihood: " + str(cf.log_likelihood_))


    if 'valid' in datasets:
        metrics = evaluate_model(cf, datasets['valid'])
        print("Valid metrics: " + str(metrics))

    if 'test' in datasets:
        metrics = evaluate_model(cf, datasets['test'], bootstrap=True)
        print("Test metrics: " + str(metrics))
    


if __name__ == '__main__':
    main()    

