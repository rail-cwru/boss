"""
Run Deletion experiment (Special entrypoint for an overnight run)
"""
import argparse
import os
import sys
import copy
import json
import pickle

from agentsystem import CoordinationGraphSystem
from common.utils import ensure_file_writable
from controller.mdp_controller import MDPController
from config import Config

GRAPHS4 = (['k', 'ring', 'spoon', 'star', '8line', '8ethane', '8ring', '8barbell', '8biring', '8duos'],
           [[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            [[0, 1], [1, 2], [2, 3], [0, 3]],
            [[0, 1], [0, 2], [0, 3], [1, 3]],
            [[0, 1], [0, 2], [0, 3]],
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
            [[0, 1], [0, 2], [0, 3], [0, 4], [4, 5], [4, 6], [4, 7]],
            [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 7]],
            [[0, 1], [1, 2], [2, 3], [0, 3], [2, 4], [4, 5], [5, 6], [6, 7], [4, 7]],
            [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7]],
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            ])


def train_trial(use_nodes, graphname, adjlist, overwrite):
    json_path = './experiment_configs/TrainPredPrey.json'
    source_data = json.load(open(json_path, 'r'))
    train_data = copy.deepcopy(source_data)

    name = modify_data_get_trialname(train_data, graphname, adjlist, use_nodes)
    for callback_config in train_data['callbacks']:
        if callback_config['name'] == 'Evaluate':
            callback_config['output_reward_file'] = "./results/{}/eval_rew.pkl".format(name)
            callback_config['save_best_mean'] = "./results/{}/policy_best_exploit".format(name)
        elif callback_config['name'] == 'SaveBest':
            callback_config['file_location'] = "./results/{}/policy_best".format(name)
        elif callback_config['name'] == 'SaveSystem':
            callback_config['file_location'] = "./results/{}/policy".format(name)

    target = './results/{}/train_results_final.pkl'.format(name)
    if not overwrite and os.path.exists(target):
        print('Target file {} exists, skipping...'.format(target))
        return
    dest = ensure_file_writable('./results/{}/train_results_final.json'.format(name))
    json.dump(train_data, fp=open(dest, 'w+'))

    config = Config(MDPController, train_data, relative_path=os.path.dirname(json_path))
    controller = MDPController(config)
    asys: CoordinationGraphSystem = controller.asys
    print('Training', asys.graph)
    result = controller.run()
    pickle.dump(result, file=open(target, 'wb+'))


def run_delete(use_nodes, graphname, adjlist, prey_ai, method, overwrite):
    json_path = './experiment_configs/DeletionExperiment.json'
    source_data = json.load(open(json_path, 'r'))
    data = copy.deepcopy(source_data)
    if '8' in graphname:
        data["episode_max_length"] = 1500
        num_agents = 8
    else:
        num_agents = 4
    name = modify_data_get_trialname(data, graphname, adjlist, use_nodes)
    trial_len = 200
    eval_len = 100
    data["episodes"] = trial_len * (num_agents * 2 + 1)
    for callback_config in data['callbacks']:
        if callback_config['name'] == 'LoadPolicySchedule':
            target = "./results/{}/policy_best_exploit.npz".format(name)
            callback_config['schedule'] = {str(trial_len*i + 1): target for i in range(0, num_agents*2 + 1)}
        elif callback_config['name'] == 'CheckpointSchedule':
            callback_config['schedule'] = {str(trial_len*i + 1): 'load' for i in range(1, num_agents*2 + 1)}
            callback_config['schedule']['1'] = 'save'
        elif callback_config['name'] == 'LearningSchedule':
            schedule = {'1': False}
            for i in range(1, num_agents * 2 + 1):
                schedule[str(trial_len*i+1)] = False
                schedule[str(trial_len*i+1+eval_len)] = True
            callback_config['schedule'] = schedule
        elif callback_config['name'] == 'DeleteAgentsSchedule':
            callback_config['schedule'] = {str((aid+1)*trial_len+1): [aid] for aid in range(0, num_agents)}
        elif callback_config['name'] == 'SimulatedDeletionSchedule':
            callback_config['schedule'] = {str((aid+num_agents+1)*trial_len+1): [aid] for aid in range(0, num_agents)}
    data['environment']['prey_ai'] = prey_ai
    data['environment']['prey_temp'] = 0.5
    data['agentsystem']['transfer_method'] = method
    target = './results/{}/deletionf_{}_{}.pkl'.format(name, prey_ai, method)

    if not overwrite and os.path.exists(target):
        print('Target {} already exists, skipping...'.format(target))
        return

    with open('./results/{}/deletionf_{}_{}.json'.format(name, prey_ai, method), 'w+') as f:
        json.dump(data, fp=f)

    config = Config(MDPController, data, relative_path=os.path.dirname(json_path))
    controller = MDPController(config)
    asys: CoordinationGraphSystem = controller.asys
    print(asys.graph, prey_ai, asys.transfer_method)
    result = controller.run()
    pickle.dump(result, file=open(target, 'wb+'))


def modify_data_get_trialname(train_data, graphname, adjlist, use_nodes):
    train_data['agentsystem']['coordination_graph'] = adjlist
    train_data['agentsystem']['use_nodes'] = use_nodes
    nodename = 'n_' if use_nodes else ''
    if '8' in graphname:
        train_data["episode_max_length"] = 1500
        train_data['environment']['map_shape'] = [64, 64]
        train_data['environment']["predators"] = ["random"] * 8
        train_data['environment']["prey"] = ["random"] * 16
        train_data["environment"]["capture_reward"] = 50
        train_data["environment"]["time_reward"] = -1
        train_data["environment"]["collision_reward"] = -5
        name = 'PredPrey8F_' + nodename + graphname
    else:
        train_data["environment"]["capture_reward"] = 5
        train_data["environment"]["time_reward"] = -0.1
        train_data["environment"]["collision_reward"] = -1
        name = 'PredPrey44_' + nodename + graphname
    return name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run deletion experiments.')
    parser.add_argument('action',
                        help='Action to do',
                        choices=['alltrial', 'simpltrial', 'train', 'delete', 'deltran', 'query', 'para'])
    parser.add_argument('--trials',
                        help='which trials to run if alltrial or simpltrial',
                        nargs='+',
                        type=int)
    parser.add_argument('--para_msg',
                        help='command for parallel run',
                        choices=['alltrial', 'simpltrial', 'train', 'delete', 'deltran', 'query'])
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrite existing files instead of skipping')
    args = parser.parse_args()
    if args.action == 'para':
        assert args.para_msg

    path = './experiment_configs/'

    trials = []
    for graphname, adjlist in zip(*GRAPHS4):
        trials.append({'use_nodes': True, 'graphname': graphname, 'adjlist': adjlist})
        trials.append({'use_nodes': False, 'graphname': graphname, 'adjlist': adjlist})

    if args.action not in ['query', 'para']:
        assert args.trials
        for arg in args.trials:
            print('arg:', int(arg), type(int(arg)))

        for arg in args.trials:
            params = trials[int(arg)]
            params['overwrite'] = args.overwrite

            if args.action in ['alltrial', 'simpltrial', 'train']:
                train_trial(**params)

            if args.action in ['alltrial', 'deltran']:
                for prey_ai in ['frozen', 'random', 'softmax_escape', 'greedy_escape']:
                    for method in ['map_to_neighbors', 'project_over_existing', 'drop_removed', 'drop_restart']:
                        params['prey_ai'] = prey_ai
                        params['method'] = method
                        run_delete(**params)
            elif args.action in ['simpltrial', 'delete']:
                params['prey_ai'] = 'softmax_escape'
                for method in ['map_to_neighbors', 'project_over_existing', 'drop_removed', 'drop_restart']:
                    params['method'] = method
                    try:
                        run_delete(**params)
                    except Exception as e:
                        print('error on run', params)
                        input('wait for input')
    else:
        print(len(trials), 'possible trials')
        # lists = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [12, 13, 18], [14, 15, 19], [11, 16, 17]]
        lists = [[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 9], [12, 18, 13], [14, 15, 19], [16, 11, 17]]
        for consumer in lists:
            trials_to_run = ' '.join([str(i) for i in consumer])
            print('run with {}'.format(trials_to_run))
            for i in consumer:
                print(trials[i])
            if args.action == 'para':
                os.system('start python src/predprey_deletion_experiment.py'
                          '{} --trials {}{}'.format(args.para_msg, trials_to_run,
                                                    ' --overwrite' if args.overwrite else ''))

