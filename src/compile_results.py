import os
import pickle as pkl
import numpy as np
import argparse

"""
This script is designed to compile results that are in multiple directories
This may occur if you run the same configuration on multiple cpus concurrently. 
To use this script, put them all in one directory. 
"""


parser = argparse.ArgumentParser(description='Compilation Script')
parser.add_argument('directory',  type=str, help='Directory')
args = parser.parse_args()

os.chdir(args.directory)

cummax = []
eval = []
lens = []
schedule = []
rt = []

with open('master_rew.txt', 'wb') as f:
    for i in os.listdir('.'):
        print(i)
        if os.path.isdir(i):

            for x in os.listdir(i):
                if 'cummax' in x and 'pkl' in x:
                    with open(i + '/' + x, 'rb') as f2:
                        arr = pkl.load(f2)
                        for arr_value in arr:
                            cummax.append(arr_value)
                            print('---------------------------------')
                            print(type(arr_value))
                            print(arr_value)
                            print('---------------------------------')

                            # raise NotImplementedError


                elif 'eval' in x and 'pkl' in x:
                    with open(i + '/' + x, 'rb') as f2:
                        arr = pkl.load(f2)
                        for arr_value in arr:
                            eval.append(arr_value)

                elif 'len' in x and 'pkl' in x:
                    with open(i + '/' + x, 'rb') as f2:
                        arr = pkl.load(f2)
                        for arr_value in arr:
                            lens.append(arr_value)

                elif 'schedule' in x and 'pkl' in x:
                    with open(i + '/' + x, 'rb') as f2:
                        arr = pkl.load(f2)
                        for arr_value in arr:
                            schedule.append(arr_value)
                elif 'rt' in x and 'pkl' in x:
                    with open(i + '/' + x, 'rb') as f2:
                        arr = pkl.load(f2)
                        for arr_value in arr:
                            rt.append(arr_value)

    cmax = np.asarray(cummax)
    evl = np.asarray(eval)
    lns = np.asarray(lens)
    sce = np.asarray(schedule)
    rt_np = np.asarray(rt)

    #print(cmax)

    for ind, i in enumerate(cummax):
        cummax[ind] = np.asarray(i, dtype=np.float64)

    print(cummax)
    print(cmax)


    cmax_array = []
    for c in cummax:
        cmax_array.append([float(i) for i in c])

    try:
        np.savetxt('cummax_master.txt', cummax)
    except:
        pass


    try:
        with open('cummax_master.pkl', 'wb') as f2:
            pkl.dump(cummax, f2)
    except:
        try:
            with open('cummax_master.pkl', 'wb') as f2:
                pkl.dump(cmax_array, f2)

        except:
            pass


    try:

        np.savetxt('eval_master.txt', evl)

        np.savetxt('len_master.txt', lns)
        np.savetxt('rt_master.txt', rt_np)
    except :
        pass


    # for i in sce:
    #     print(i)

    with open('len_master.pkl', 'wb') as f2:
        pkl.dump(lens, f2)

    with open('eval_master.pkl', 'wb') as f2:
        pkl.dump(eval, f2)

    with open('schedule_master.pkl', 'wb') as f2:
        pkl.dump(schedule, f2)

    with open('rt_master.pkl', 'wb') as f2:
        pkl.dump(rt, f2)

    print('Num Examples:', len(cummax))
