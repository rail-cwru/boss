import os
import sys
import bisect
import matplotlib.pyplot as plt
import numpy as np
# Specify parametesr
alg = 'reinforce'
env = 'cartpole'
experience_interval = 100
use_rolling_avg = False

data_dirs = {
    #'experiments\\hyperparameter_tuning\\results\\{}\\tb\\{}\\'.format(env, alg) : 'Tuned Baseline',
    #'experiments\\hyperparameter_tuning\\results\\{}\\rb\\{}\\'.format(env, alg) : 'Random Avg Baseline',
    'experiments\\hyperparameter_tuning\\results\\{}\\ftl_mf\\{}\\'.format(env, alg) : 'FTL',
    'experiments\\hyperparameter_tuning\\results\\{}\\pbt\\{}\\'.format(env, alg) : 'PBT',
    'experiments\\hyperparameter_tuning\\results\\{}\\ib\\{}\\'.format(env, alg) : 'Ideal Tuning Baseline'
}

max_reward = {
    'cartpole': 500.0
}

# constant File names
eval_file = 'eval.csv'
train_file = 'train.csv'
pbt_additional = 'eval_experience.csv'

# Constant Columns
experience_col = 2
avg_train_reward_col = 3
avg_eval_reward_col = 2

def read_default_files(directory: str, exp_file_name: str, reward_file_name: str):
    experience = []
    reported_rewards = []

    # Read training experience
    reward_history = []
    data_file = directory + '\\' + exp_file_name
    with open(data_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                exp_val = float(row_split[experience_col])
                if len(experience) > 0:
                    exp_val += experience[-1]
                experience.append(exp_val)

                if use_rolling_avg:
                    reward_val = float(row_split[avg_train_reward_col])
                    reward_history.append(reward_val)

                    reported_reward = np.mean(reward_history[-100:])
                    reported_rewards.append(reported_reward)
            row_index += 1

    # Read evaluation rewards, if required
    if not use_rolling_avg:
        data_file = directory + '\\' + reward_file_name
        with open(data_file, mode='r') as csv_file:
            row_index = 0
            for row in csv_file:
                row_split = row.split(',')     
                # Skip header      
                if row_index != 0:
                    reward_val = float(row_split[avg_eval_reward_col])
                    if row_index == 1:
                        reported_reward = reward_val
                    else:
                        reported_reward = max(reward_val, reported_rewards[-1])
                    reported_rewards.append(reported_reward)
                row_index += 1

    return np.array(experience), np.array(reported_rewards)

def read_pbt_files(directory: str):
    # Identify idenicies off of files
    result_files = [x[2] for x in os.walk(directory)][0]
    file_indicies = []
    for result_file in result_files:
        if pbt_additional in result_file:
            additional_file_name = result_file
        elif '_' in result_file:
            index = os.path.splitext(result_file)[0].split('_')[-1]
            file_indicies.append(int(index))
        elif result_file == eval_file or result_file == train_file:
            file_indicies.append(0)
            
    file_indicies = list(set(file_indicies))

    # Read additional file
    eval_experiences = []
    data_file = directory + '\\' + additional_file_name
    with open(data_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                exp_val = float(row_split[1])
                eval_experiences.append(exp_val)
            row_index += 1
    
    # Read files
    index_data = {}
    for index in file_indicies:
        # Create file names
        if index == 0:
            index_train_file = train_file
            index_eval_file = eval_file
        else:
            index_train_file = '{0}_{2}{1}'.format(*os.path.splitext(train_file) + (index,))
            index_eval_file = '{0}_{2}{1}'.format(*os.path.splitext(eval_file) + (index,))
        experience, reward = read_default_files(directory, index_train_file, index_eval_file)

        index_data[index] = (np.array(experience), np.array(reward))

    # Average across
    index_max_reward = []
    sum_experience = []
    for i, eval_experience in enumerate(eval_experiences):
        exp_limit = (i + 1) * experience_interval
        index_r_vals = []
        index_e_vals = []
        for e, r in index_data.values():
            index = bisect.bisect(e, exp_limit)
            index_e_vals.append(e[index])
            index_r_vals.append(r[index])
        
        index_max_reward.append(np.max(index_r_vals))
        sum_experience.append(np.sum(index_e_vals) + eval_experience)

    return np.array(sum_experience), np.array(index_max_reward)

def read_rb_files(directory: str):
    # Identify idenicies off of files
    result_files = [x[2] for x in os.walk(directory)][0]
    file_indicies = []
    for result_file in result_files:
        if '_' in result_file:
            index = os.path.splitext(result_file)[0].split('_')[-1]
            file_indicies.append(int(index))
        elif result_file == eval_file or result_file == train_file:
            file_indicies.append(0)
    file_indicies = list(set(file_indicies))
    
    # Read files
    index_data = {}
    for index in file_indicies:
        # Create file names
        if index == 0:
            index_train_file = train_file
            index_eval_file = eval_file
        else:
            index_train_file = '{0}_{2}{1}'.format(*os.path.splitext(train_file) + (index,))
            index_eval_file = '{0}_{2}{1}'.format(*os.path.splitext(eval_file) + (index,))
        experience, reward = read_default_files(directory, index_train_file, index_eval_file)

        index_data[index] = (np.array(experience), np.array(reward))

    # If max reward is found extend experience
    if not use_rolling_avg:
        index_max_e = max([max(e) for e, r in index_data.values()])
        for index in file_indicies:
            e, r = index_data[index]
            if r[-1] >= max_reward[env]:
                e = np.append(e, [index_max_e])
                r = np.append(r, [max_reward[env]])
                index_data[index] = (e, r)

    # Average across
    index_minimax_e = min([max(e) for e, r in index_data.values()])
    avg_reward = []
    avg_experience = []
    for x_val in np.arange(experience_interval, index_minimax_e, experience_interval):
        index_vals = []
        for e, r in index_data.values():
            index = bisect.bisect(e, x_val)
            if index < r.shape[0]:
                index_vals.append(r[index])
            else:
                index_vals.append(r[-1])

        avg_reward.append(np.mean(index_vals))
        avg_experience.append(x_val)

    return np.array(avg_experience), np.array(avg_reward)

# Read data files in and create trace
data_across_seeds = {}
ordered_data_dirs = list(data_dirs.keys())
ordered_data_dirs.sort()
for data_dir in ordered_data_dirs:
    label = data_dirs[data_dir]
    seed_data = {}

    # Find all seed directories
    seed_dirs = [x[0] for x in os.walk(data_dir) if x[0] != data_dir]

    for seed_dir in seed_dirs:
        str_seed = seed_dir[-1]

        if 'pbt' in seed_dir:
            experience, reward = read_pbt_files(seed_dir)
        elif 'rb' in seed_dir:
            experience, reward = read_rb_files(seed_dir)
        else:
            experience, reward = read_default_files(seed_dir, train_file, eval_file)

        # Create trace
        seed_data[str_seed] = experience, reward

    data_across_seeds[label] = seed_data

# Average across seeds
data_traces = []
for label, seed_data in data_across_seeds.items():
    # If max reward is found extend experience
    label_max_e = max([max(e) for e, r in seed_data.values()])
    if not use_rolling_avg:
        for seed in seed_data.keys():
            e, r = seed_data[seed]
            if r[-1] >= max_reward[env]:
                e = np.append(e, [label_max_e])
                r = np.append(r, [max_reward[env]])
                seed_data[seed] = (e, r)

    x = []
    y = []
    label_minimax_e = min([max(e) for e, r in seed_data.values()])
    for x_val in np.arange(experience_interval, label_max_e, experience_interval):
        seed_vals = []
        for e, r in seed_data.values():
            index = bisect.bisect(e, x_val)
            if index < r.shape[0]:
                seed_vals.append(r[index])
            else:
                seed_vals.append(r[-1])

        y.append(np.mean(seed_vals))
        x.append(x_val)

    data_traces.append((x, y, label))

# Create graph
for x, y, label in data_traces:
    plt.plot(x, y, label=label)
plt.xlabel('Experience')
plt.ylabel('Expected Reward')
plt.legend(loc="lower right")
plt.show()