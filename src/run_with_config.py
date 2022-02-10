"""
The standard provided entrypoint to run the framework with a specified config file.
Takes two arguments:
    1) Path to the config file
    2) A boolean indicating if the analysis is to be performed offline
        This value defaults to False, indicating online analysis

    Results are saved in both .txt and .pkl files as well as printed to standard out
"""
import os
os.environ['PYTHONHASHSEED'] = str('0')
import json
import argparse
import time
import datetime
import matplotlib
matplotlib.use('Agg')
from results_manager import ResultsManager

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Uncomment to use CPU only

# TODO have controller imported via config name to finally unify classes
from controller import MDPController
from controller.offline_controller import OfflineController
from config import Config

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the RL framework with MDPController and a preset configuration.')
    parser.add_argument('config_file',
                        help='A path to the experiment configuration JSON file.')
    parser.add_argument('offline', help='Is learning offline?', nargs='?', default=False, type=str2bool)
    parser.add_argument('num_analyses', help='How many analyses to perform', nargs='?', default=10, type=int)

    current_date = datetime.datetime.now()

    args = parser.parse_args()
    num_analyses = args.num_analyses
    results_manager = ResultsManager(args.offline)

    json_path = args.config_file
    data = json.load(open(json_path, 'r'))
    path = os.path.dirname(json_path)

    for iteration in range(num_analyses):
        print('---------------------------------------------')
        print(args.config_file)

        if args.offline:
            config = Config(OfflineController, data, relative_path=path)
            controller = OfflineController(config, iteration, current_date)

        else:
            config = Config(MDPController, data, relative_path=path)
            controller = MDPController(config)

        is_boss = hasattr(config, "sampler") and config.sampler.name == 'BOSS'

        if hasattr(config, 'samples_name') and config.samples_name is not '':
            save_name = config.samples_name
        else:
            raise ValueError('No Name In Config File')

        if iteration == 0:
            dir_name = controller.make_results_directory(current_date, save_name)
            results_manager.set_dir_name(dir_name)

        if args.offline and config.sampler.name == 'FlattenedPolledSampler' and iteration == 0:
            controller.save_flattened_hierarchy(dir_name, save_name, config)

        collect_novel = hasattr(config, 'novel_states_count') and config.novel_states_count

        # Run analysis
        start = time.perf_counter()
        controller.run()
        end = time.perf_counter()

        results_manager.append_times(end-start)
        results_manager.after_run(config, controller, save_name, is_boss, collect_novel, iteration)



