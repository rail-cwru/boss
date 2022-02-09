import os
import numpy as np
from typing import List

class Logger(object):
    def __init__(self, result_file: str):
        # Create directory if does not exist
        directory = os.path.dirname(result_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create file names
        self.alg_definition_file = '{}_{}.csv'.format(result_file, 'alg_def')
        self.alg_selection_file = '{}_{}.csv'.format(result_file, 'alg_sel')
        
        # Initialize files
        self._initialize_files()

    def _initialize_files(self):
        # Delete previous version
        if os.path.exists(self.alg_definition_file):
            os.remove(self.alg_definition_file)

        if os.path.exists(self.alg_selection_file):
            os.remove(self.alg_selection_file)

    def reset(self, algorithms):
        # Reinitialize files
        self._initialize_files()

        # Write algorithm definition file
        self._finalize_alg_def_file(self.alg_definition_file, algorithms)

        # Write header to selection file
        column_header = ['Episode Number', 'Algorithm ID']
        with open(self.alg_selection_file, mode='w') as f:
            # Write header
            header = ','.join(column_header)
            f.write(header + '\n')

    def log_selection(self, episode_num: int, alg_index: int):
        with open(self.alg_selection_file, mode='a') as f:
            row = [str(episode_num), str(alg_index)]
            row_str = ','.join(row)
            f.write(row_str + '\n')

    def _finalize_alg_def_file(self, result_file: str, algorithms):
        # Identify parameter names
        alg_param_names =  [param.name for param in algorithms[0].algorithm.get_class_config()]
        policy_sampler_param_names = [param.name for param in algorithms[0].policy.sampler.get_class_config()]

        # Write algorithm configurations
        column_header = ['Algorithm ID'] + alg_param_names + policy_sampler_param_names
        with open(result_file, mode='w') as f:
            # Write header
            header = ','.join(column_header)
            f.write(header + '\n')

            for i_alg, offline_learner in enumerate(algorithms):
                # Get parameter values
                alg_row = [str(getattr(offline_learner.algorithm.config, param)) for param in alg_param_names]
                ps_row = [str(getattr(offline_learner.policy.sampler.config, param)) for param in policy_sampler_param_names]
                row = alg_row + ps_row
                row.insert(0, str(i_alg))

                # Write row
                row_str = ','.join(row)
                f.write(row_str + '\n')

