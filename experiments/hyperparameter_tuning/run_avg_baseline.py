import os
import sys
import json
import argparse
from copy import deepcopy

sys.path.append(os.getcwd() + '\\src')
from controller import MDPController
from config import Config

def read_config_values(config_file):
    config_values = {}

    with open(config_file, mode='r') as f:
        row_index = 0
        for row in f:
            row_split = row.rstrip().split(',')     
            # Skip header      
            if row_index == 0:
                if 'qlearning' in config_file:
                    alg_keys = row_split[1:7]
                    policy_sampler_keys = row_split[7:]
                else:
                    alg_keys = row_split[1:6]
                    policy_sampler_keys = row_split[6:]
            else:
                alg_id = int(row_split[0])
                if 'qlearning' in config_file:
                    alg_values = row_split[1:7]
                    policy_sampler_values = row_split[7:]

                else:
                    alg_values = row_split[1:6]
                    policy_sampler_values = row_split[6:]

                # Convert types
                for i in range(len(alg_values)):
                    if alg_values[i] == 'True' or alg_values[i] == 'False':
                        alg_values[i] = bool(alg_values[i])
                    elif '.' in alg_values[i] or 'e' in alg_values[i]:
                        alg_values[i] = float(alg_values[i])
                    else:
                        alg_values[i] = int(alg_values[i])

                for i in range(len(policy_sampler_values)):
                    if policy_sampler_values[i] == 'True' or policy_sampler_values[i] == 'False':
                        policy_sampler_values[i] = bool(policy_sampler_values[i])
                    elif '.' in policy_sampler_values[i] or 'e' in policy_sampler_values[i]:
                        policy_sampler_values[i] = float(policy_sampler_values[i])
                    else:
                        policy_sampler_values[i] = int(policy_sampler_values[i])

                # Create two dicts
                alg_config = dict(zip(alg_keys, alg_values))
                policy_sampler_config = dict(zip(policy_sampler_keys, policy_sampler_values))

                config_values[alg_id] = (alg_config, policy_sampler_config)
            row_index += 1

    return config_values

def main():
    parser = argparse.ArgumentParser(description='Run the RL framework with MDPController and a preset configuration.')
    parser.add_argument('config_file', help='A path to the experiment configuration JSON file.')
    parser.add_argument('config_values_file', help='A path to the experiment configuration value CSV file.')
    args = parser.parse_args()

    initial_config_file = args.config_file
    config_value_file = args.config_values_file
    seeds = [3]

    # Read config values
    alg_values = read_config_values(config_value_file)

    # Read initial config
    data = json.load(open(initial_config_file, 'r'))
    
    for seed in seeds:
        for alg_id in alg_values.keys():
            if alg_id < 8:
                continue
            
            # Update with seed format
            new_data = deepcopy(data)

            # Replace values
            new_data['seed'] = seed
            alg_config, policy_sampler_config = alg_values[alg_id]
            for k,v in alg_config.items():
                new_data['algorithm'][k] = v
            for k,v in policy_sampler_config.items():
                new_data['policy']['policy_sampler'][k] = v

            new_config = Config(MDPController, new_data)
            if 'Evaluate' in new_config.callbacks.keys():
                new_config.callbacks['Evaluate'].output_reward_file = new_config.callbacks['Evaluate'].output_reward_file.format(seed, alg_id)
            if 'SaveReward' in new_config.callbacks.keys():
                new_config.callbacks['SaveReward'].file_location = new_config.callbacks['SaveReward'].file_location.format(seed, alg_id)

            controller = MDPController(new_config)
            print(controller.run())

if __name__ == '__main__':
    main()