import os
import pickle as pkl
import numpy as np
import json
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import argparse


"""
This is a plotting script designed to plot results from the offline RL framework. 
The JSON file is expected to contain the following fields:
    coeff: a 1d list of size 2 designating the data to plot. 
        Ex: ["cummax_", "len_"]
    title: The of the plot
        Ex: "Taxi: Rewards vs. Samples Collected"
    save_title: Name to save the plot
        Ex: "6_taxi.png"
    to_plot: a 2d list determining which directories to include in the plot 
        Each entry contains the path to the directory, the suffix of the file 
            and the legend name for the plot
        Ex:     ["taxi_no_numpy/taxi_buf", "taxi_buf_master", "BUF"],
                ["taxi_no_numpy/taxi_tdf", "taxi_tdf_master", "TDF"],
                ["taxi_no_numpy/taxi_huf", "taxi_huf_master", "HUF"],
                ["taxi_no_numpy/taxi_polled", "taxi_polled_master", "Polled"],
                ["taxi_no_numpy/taxi_multi_10", "master", "BOSS"]]
                
            
"""

parser = argparse.ArgumentParser(description='Plotting Script')
parser.add_argument('json_file',  type=str, help='path to json')
args = parser.parse_args()

with open(args.json_file) as f:
    json_file = json.load(f)

coeff = json_file['coeff']
title = json_file['title']
save_title = json_file['save_title']

plt.clf()
kl = False

if 'kl' in coeff[1] or 'kl' in coeff[0]:
    kl = True

linestyle_tuple = [
                        ('solid',      (0, ())),
                        ('dotted', (0, (1, 1))),
                        ('dashdotted', (0, (3, 5, 1, 3))),
                        ('dashed', (0, (5, 3))),
                        ('dashdotdotted', (0, (3, 2, 1, 2, 1, 2))),

                        ('loosely dashed', (0, (5, 10))),

                        ('densely dotted', (0, (1, 1))),

                        ('loosely dashdotted', (0, (3, 10, 1, 10))),
                        ('densely dashed', (0, (5, 1))),

                        ('densely dashdotted', (0, (3, 1, 1, 1))),
                        ('loosely dotted', (0, (1, 10))),

                        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

count = 0
min_ax = 1e6

for i in json_file['to_plot']:

    X = i[0] + '/' + coeff[1] + i[1] + '.pkl'
    Y = i[0] + '/' + coeff[0] + i[1] + '.pkl'

    infile_x = open(X, 'rb')
    infile_y = open(Y, 'rb')

    X = pkl.load(infile_x)
    Y = pkl.load(infile_y)

    if type(X[0]) is int:
        X = [X]
        Y = [Y]

    if len(X) > 1:

        if np.mean(X, axis=0)[-1] < min_ax:
            min_ax = np.mean(X, axis=0)[-1]

        if not kl:

            cumsum = [x for x in np.maximum.accumulate(np.mean(Y, axis=0))]
            plt.plot(np.mean(X, axis=0), cumsum, label=i[-1],
                    linestyle=linestyle_tuple[count][-1], linewidth=3)
        else:
            cumsum = [x for x in np.minimum.accumulate(np.mean(Y, axis=0))]
            plt.plot(np.mean(X, axis=0), cumsum, label=i[-1],
                     linestyle=linestyle_tuple[count][-1], linewidth=3)
    else:
        if X[0][-1] < min_ax:
            min_ax = X[0][-1]
        if not kl:
            plt.plot(X[0],[x for x in np.maximum.accumulate(Y[0])], label=i[-1],
                 linestyle=linestyle_tuple[count][-1], linewidth=3)
        else:
            plt.plot(X[0], [x for x in np.minimum.accumulate(Y[0])], label=i[-1],
                     linestyle=linestyle_tuple[count][-1], linewidth=3)

    plt.legend()
    plt.title(title, fontsize=18)

    count += 1

plt.grid(which="both", linewidth=0.5)

plt.ylabel('Reward', fontsize=16)
if 'rt' in coeff[1]:
    plt.xlabel('Runtime (seconds)', fontsize=16)
elif 'rt' in coeff[0]:
    plt.ylabel('Runtime (seconds)', fontsize=16)
if 'len' in coeff[1]:
    plt.xlabel('Samples Collected', fontsize=16)
    plt.xlim([0, min_ax])

plt.savefig(json_file['save_title'])

print("Saved: ", json_file['save_title'])
