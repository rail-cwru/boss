import os
import pickle as pkl
import numpy as np
import json
import matplotlib as mpl
mpl.use('Agg')
import os
from matplotlib import pyplot as plt
import argparse

"""
This is a script aimed to plot the schedule of the BOSS Sampler. 
It takes the following arguments:
    file_name: path to schedule file
    plot_name: name to save plot as
    plot_title: Title of generated plot
    len_file: path to length file (for the X-axis)
    
"""


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


parser = argparse.ArgumentParser(description='Plotting Script')
parser.add_argument('file_name',  type=str, help='name of file')
parser.add_argument('plot_name',  type=str, help='name of plot')
parser.add_argument('plot_title',  type=str, help='title of plot')
parser.add_argument('len_file',  type=str, help='path to dir', default='')
args = parser.parse_args()


f = args.file_name
plot_name = args.plot_name
    #"jump_exp_100_opt-9-8-2021_2207/schedule_jump_exp_100_opt_master.pkl"

with open(f, 'rb') as file:
    s = pkl.load(file)

sampler_list = ['HUF', 'Polled', 'TDF_1', 'TDF_2', 'TDF_3', 'TDF_4', 'BUF_1', 'BUF_2',
                'BUF_3', 'BUF_4', "BUF_5", "BUF_6", "BUF_7", 'TDF_5', "TDF_6"]

sampler_dict = {}
len_run = len(s[0])

for i in sampler_list:
    sampler_dict[i] = np.zeros(len_run)

plot_set = set()
l_c = 0


if args.len_file:
    len_file = args.len_file

    with open(len_file,'rb') as f:
        lens = pkl.load(f)

else:
    ## Find length arr
    dir = args.file_name.split('/')[0]
    len_file = ''

    for i in os.walk(dir):
        for x in i[-1]:
            print(x)
            if 'len' in x and 'pkl' in x:
                len_file = x

    if not len_file:
        raise FileNotFoundError('No Len File')

    with open(dir + '/' + len_file,'rb') as f:
        lens = pkl.load(f)

linestyle_tuple = [

                        ('dotted', (0, (1, 1))),
                        ('dashdotted', (0, (3, 5, 1, 5))),
                        ('dashed', (0, (5, 5))),
                        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),


                        ('loosely dashed', (0, (5, 10))),

                        ('densely dotted', (0, (1, 1))),

                        ('loosely dashdotted', (0, (3, 10, 1, 10))),
                        ('densely dashed', (0, (5, 1))),

                        ('densely dashdotted', (0, (3, 1, 1, 1))),
                        ('loosely dotted', (0, (1, 10))),

                        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

for i in range(len_run):
    count_dict = {}
    l_c = 0
    for j in s:
        if j[i] not in count_dict:
            count_dict[j[i]] = 1
        else:
            count_dict[j[i]] += 1
        l_c += 1

    for c in count_dict:
        sampler_dict[c][i] = count_dict[c]/l_c
        plot_set.add(c)

plt.clf()
count = 0
for pl in plot_set:
    plt.plot(lens[0], sampler_dict[pl], label=pl, linestyle=linestyle_tuple[count][-1], linewidth=3)
    count += 1

plt.grid(which="both", linewidth=0.5)
plt.legend()
plt.title(args.plot_title)
plt.xlabel('Total Samples Collected')
plt.ylabel('Percentage of times sampler was chosen')
plt.savefig(plot_name)


## plot moving averages

window = 3
plt.clf()
count = 0
for pl in plot_set:
    plt.plot(lens[0][:-1*(window - 1)], moving_average(sampler_dict[pl], window), label=pl,
             linestyle=linestyle_tuple[count][-1], linewidth=3)
    count += 1

plt.grid(which="both", linewidth=0.5)
plt.legend()
plt.title(args.plot_title + ' MA')
plt.xlabel('Total Samples Collected')
plt.ylabel('Percentage of times sampler was chosen')
plt.savefig(plot_name+ '_ma')

print('Saved: ', plot_name+ '_ma')
