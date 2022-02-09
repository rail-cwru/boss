import os
import sys
import json
import argparse
from copy import deepcopy

sys.path.append(os.getcwd() + '\\src')
from controller import MDPController
from config import Config

def main():
    parser = argparse.ArgumentParser(description='Run the RL framework with MDPController and a preset configuration.')
    parser.add_argument('config_file', help='A path to the experiment configuration JSON file.')
    args = parser.parse_args()

    initial_config_file = args.config_file
    seeds = [2]

    # Read initial config
    data = json.load(open(initial_config_file, 'r'))
    
    for seed in seeds:
        # Update with seed format
        new_data = deepcopy(data)

        # Replace values
        new_data['seed'] = seed
        seed_config = Config(MDPController, new_data)

        if 'Evaluate' in seed_config.callbacks.keys():
            seed_config.callbacks['Evaluate'].output_reward_file = seed_config.callbacks['Evaluate'].output_reward_file.format(seed)
        if 'SaveReward' in seed_config.callbacks.keys():
            seed_config.callbacks['SaveReward'].file_location = seed_config.callbacks['SaveReward'].file_location.format(seed)
        if 'TuneHyperparameters' in seed_config.callbacks.keys():
            seed_config.callbacks['TuneHyperparameters'].tuning_strategy.file_location = seed_config.callbacks['TuneHyperparameters'].tuning_strategy.file_location.format(seed)

        controller = MDPController(seed_config)
        print(controller.run())

if __name__ == '__main__':
    main()