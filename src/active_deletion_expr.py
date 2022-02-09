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

EVAL_LEN = 100
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


def trial(use_nodes, graphname, adjlist, method, overwrite):
    json_path = './experiment_configs/GridBattleActiveDeletion.json'
    source_data = json.load(open(json_path, 'r'))
    data = copy.deepcopy(source_data)
    data["episode_max_length"] = 1500 if '8' in graphname else 1000
    name = modify_data_get_trialname(data, graphname, adjlist, use_nodes)
    data["episodes"] = EVAL_LEN

    chkp_callback = {'name': 'CheckpointSchedule', 'schedule': {str(i+1): 'load' for i in range(0, data['episodes'])}}
    chkp_callback['schedule']['1'] = 'save'
    data['callbacks'] = [chkp_callback] + data['callbacks']
    for callback_config in data['callbacks']:
        # TODO figure this out after discussion
        # elif callback_config['name'] == 'SimulatedDeletionSchedule':
        #     callback_config['schedule'] = {str((aid+num_agents+1)*trial_len+1): [aid] for aid in range(0, num_agents)}
        pass
    data['agentsystem']['transfer_method'] = method
    target = ensure_file_writable('./results/{}/activedel_{}.pkl'.format(name, method))

    if not overwrite and os.path.exists(target):
        print('Target {} already exists, skipping...'.format(target))
        return

    with open('./results/{}/activedel_{}.json'.format(name, method), 'w+') as f:
        json.dump(data, fp=f)

    config = Config(MDPController, data, relative_path=os.path.dirname(json_path))
    controller = MDPController(config)
    asys: CoordinationGraphSystem = controller.asys
    print(asys.graph, asys.transfer_method)
    result = controller.run()
    pickle.dump(result, file=open(target, 'wb+'))


def modify_data_get_trialname(train_data, graphname, adjlist, use_nodes):
    train_data['agentsystem']['coordination_graph'] = adjlist
    train_data['agentsystem']['use_nodes'] = use_nodes
    nodename = 'n_' if use_nodes else ''
    if '8' in graphname:
        train_data["episode_max_length"] = 1500
        train_data['environment']['map_shape'] = [64, 64]
        train_data['environment']["teams"][0]["init_positions"] = ["random"] * 8
        train_data['environment']["teams"][1]["init_positions"] = ["random"] * 8
        name = 'GridBattle8v8_' + nodename + graphname
    else:
        train_data["episode_max_length"] = 1000
        train_data['environment']['map_shape'] = [32, 32]
        train_data['environment']["teams"][0]["init_positions"] = ["random"] * 4
        train_data['environment']["teams"][1]["init_positions"] = ["random"] * 4
        name = 'GridBattle4v4_' + nodename + graphname
    return name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run deletion experiments.')
    parser.add_argument('action',
                        help='Action to do',
                        choices=['delete', 'deltran', 'query', 'para'])
    parser.add_argument('--trials',
                        help='which trials to run if alltrial or simpltrial',
                        nargs='+',
                        type=int)
    parser.add_argument('--para_msg',
                        help='command for parallel run',
                        choices=['delete', 'deltran', 'query'])
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

            if args.action in ['alltrial', 'deltran']:
                for method in ['map_to_neighbors', 'project_over_existing', 'drop_removed', 'drop_restart']:
                    params['method'] = method
                    trial(**params)
            elif args.action in ['simpltrial', 'delete']:
                for method in ['map_to_neighbors', 'project_over_existing', 'drop_removed', 'drop_restart']:
                    params['method'] = method
                    try:
                        trial(**params)
                    except Exception as e:
                        print('error on run', params)
                        # input('wait for input')
                        raise e
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
                os.system('start python src/active_deletion_expr.py'
                          '{} --trials {}{}'.format(args.para_msg, trials_to_run,
                                                    ' --overwrite' if args.overwrite else ''))

