import json
import pickle

import numpy as np

from config import Config
from controller.mdp_controller import MDPController
from environment import GridWorld
from environment import Cartpole
from environment.MountainCarContinuous import MountainCarContinuous


def test_gridworld():
    with open('../experiment_configs/environment/gridworld.json') as f:
        edict = json.load(f)
        environment = type('environment', (object,), edict)
        config = type('config', (object,), {'environment' : environment})
        gw = GridWorld(config)
        print(gw.observe())
        for i in range(100):
            # input("i")
            actiondomain = gw.action_domain[0]
            moveIndex = actiondomain.index_for_name('move')
            actions = {}
            for agent in gw.agent_id_list:
                act_range = actiondomain.items[0].range
                action = np.random.randint(act_range.start, act_range.stop, size=actiondomain.shape)
                actions[agent] = action
            # print(actions)
            print(gw.update(actions=actions))
            print(gw.terminated_count)
            gw.visualize()


def test_cartpole():
    with open('../experiment_configs/environment/cartpole.json') as f:
        edict = json.load(f)
        environment = type('environment', (object,), edict)
        config = type('config', (object,), {'environment' : environment})
        env = Cartpole(config)
        for i in range(100):
            # input("i")
            actiondomain = env.action_domain[0]
            actions = {}
            for agent in env.agent_id_list:
                act_range = actiondomain.items[0].range
                action = np.random.randint(act_range.start, act_range.stop, size=actiondomain.shape)
                actions[agent] = action
            print(actions)
            print(env.update(actions=actions))
            # print(env.terminated_count)
            env.visualize()

def test_mountainCarContinuous():
    with open('../experiment_configs/environment/mountaincar_continuous.json') as f:
        edict = json.load(f)
        environment = type('environment', (object,), edict)
        config = type('config', (object,), {'environment' : environment})
        env = MountainCarContinuous(config)
        for i in range(100):
            # input("i")
            actiondomain = env.action_domain[0]
            actions = {}
            for agent in env.agent_id_list:
                action = np.random.rand(1)
                action = action * 2 -1
                actions[agent] = action
            print(actions)
            print(env.update(actions=actions))
            # print(env.terminated_count)
            env.visualize()


def test_predatorPrey():
    config = Config('../experiment_configs/debug_cgql.json')
    controller = MDPController(config)
    observation_request = controller.asys.observe_request()
    observation = controller.env.observe(observation_request)
    reference = pickle.load(open('./predprey_cgql_debug_obs1.pkl', mode='rb'))
    for k, arr in observation.items():
        ref = reference[k]
        diff = arr - ref.flatten()
        diff_sum = diff.sum()
        assert diff_sum < 1e-5
    env = controller.env
    for i in range(5):
        actions = {agent: np.array([np.random.randint(4)]) for agent in env.agent_id_list}
        print(actions)
        print(env.update(actions))


if __name__ == '__main__':
    test_gridworld()

