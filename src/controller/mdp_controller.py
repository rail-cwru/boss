"""
An example Controller.
"""
from collections import OrderedDict
from typing import Any, Dict, List, Callable
import numpy as np
import copy
import os
from callbacks import CallbackImpl
from common import set_initial_seed, get_seed_state, set_seed_state
from common.properties import Properties
from config.moduleframe import AbstractModuleFrame
from config.config import ConfigItemDesc, ConfigDesc, ConfigListDesc
from environment import Environment
from sampler import PolledSampler
from agentsystem import AgentSystem, HierarchicalSystem
from config import Config, checks
from common.trajectory import TrajectoryCollection
from timeit import default_timer as timer

class MDPController(AbstractModuleFrame):
    """
    Runs environment and learns agentsystem in an online manner, over episodes.
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return [
            # TODO add default configs
            ConfigItemDesc('episodes', checks.positive_integer, 'Number of episodes to run'),
            ConfigItemDesc('episode_max_length', checks.positive_integer, 'Max length of any episode'),
            ConfigDesc('environment', 'Environment config', default_configs=[]),
            ConfigDesc('agentsystem', 'Agentsystem config', default_configs=[]),
            ConfigDesc('policy', 'Policy config', default_configs=[]),
            ConfigDesc('algorithm', 'Algorithm config', default_configs=[]),
            ConfigListDesc('callbacks', 'List of callback configs', default_configs=[]),
            ConfigItemDesc('num_trajectories', checks.nonnegative_integer,
                           'How many past trajectories to keep in memory.'
                           'Trajectories may take up a lot of RAM if kept in memory.'
                           '0 to keep all trajectories (use only if you are certain).'
                           '\nDefault: 5.', default=5, optional=True),
            ConfigItemDesc('seed', checks.nonnegative_integer, 'Seed for random variables.\n Default: 2', default=2,
                           optional=True),
            ConfigItemDesc('save_traj', checks.boolean, 'Accumulate all primitive actions', default=False,
                           optional=True),
            ConfigItemDesc('samples_name', checks.string, 'Name for saving samples for future testing',
                           default='', optional=True)
        ]

    @classmethod
    def properties(cls) -> Properties:
        return Properties()

    def __init__(self, config: Config):
        # Set seed
        self.seed = config.seed
        set_initial_seed(config.properties.pytorch, config.seed)

        # Number of episodes
        self.episodes = config.episodes

        # Max episode step count. 0 if you want to wait for termination
        self.episode_max_length = config.episode_max_length

        self.eval_max_length = 50

        # How many trajectories to keep in memory. 0 to keep all.
        # TODO better storage in future: memmap? caching? we would need better storage for "all trajectories"
        #      throwing out past trajectories is a crutch that shouldn't be relied on.
        self.num_trajectories = config.num_trajectories

        # Instantiate environment
        print('Creating environment...')
        self.env: Environment = config.environment.module_class(config)
        print('...Created environment [{}].'.format(config.environment.name))

        print('Creating agentsystem...')
        self.asys: AgentSystem = config.agentsystem.module_class(self.env.agent_class_map,
                                                                 self.env.agent_class_action_domains,
                                                                 self.env.agent_class_observation_domains,
                                                                 self.env.get_auxiliary_info(),
                                                                 config)
        print('...Created agentsystem [{}].'.format(config.agentsystem.name))

        # I REALLY feel like I've cursed myself by writing this code but here it is.
        print('Hooking in Callbacks in order applied...')
        active_callback_fns: Dict[str, List[Callable]] = OrderedDict()
        self.callbacks = []
        for callback_config in config.callbacks.values():
            #print('...Found Callback [{}]'.format(callback_config.name))
            self.callbacks.append(callback_config.module_class(self, config))
        for method_key in CallbackImpl.keys:
            for callback in self.callbacks:
                if callback.implements(method_key):
                    if method_key not in active_callback_fns:
                        active_callback_fns[method_key] = []
                    active_callback_fns[method_key].append(getattr(callback, method_key))
        for method_key, func_list in active_callback_fns.items():
            if method_key != 'finalize':
                if method_key in ['on_observe', 'on_action', 'on_update']:
                    def replacement_function(*args, _func_list=func_list, **kwargs):
                        result = _func_list[0](*args, **kwargs)
                        for func in _func_list[1:]:
                            result = func(*result, **kwargs)
                        return result
                else:
                    def replacement_function(_func_list=func_list):
                        [func() for func in _func_list]
            else:
                def replacement_function(_func_list=func_list):
                    return {func.__self__.__class__.__name__: func() for func in _func_list}
            setattr(self, method_key, replacement_function)

        self.config = config
        if hasattr(config, 'save_traj'):
            self.save_traj = config.save_traj
            if self.save_traj:
                self.all_traj = []
        else:
            self.save_traj = False

        self.episode_num = 0
        self.episode_step = 0
        self.curr_trajectory: TrajectoryCollection = None
        self.episode_trajectories: List[TrajectoryCollection] = []

        # For inter-callback communication which affects controller behavior
        self.flags = ControllerFlags()

    def get_checkpoint(self):
        # TODO (future) metaclassing to generate on-the-fly subclasses of some parent Checkpoint class
        # TODO (optimize) more efficient retrieval of elements to avoid copying unnecessarily (?)
        return {'class': self.__class__,
                'env': copy.deepcopy(self.env),
                'asys': copy.deepcopy(self.asys)}

    def set_from_checkpoint(self, checkpoint):
        assert checkpoint['class'] == self.__class__
        self.env = copy.deepcopy(checkpoint['env'])
        self.asys = copy.deepcopy(checkpoint['asys'])

    def get_seed_state(self):
        use_pytorch = self.config.properties.pytorch
        return get_seed_state(use_pytorch, self.env)

    def set_seed_state(self, seed_state):
        use_pytorch = self.config.properties.pytorch
        set_seed_state(use_pytorch, self.env, seed_state)

    def run(self):
        # print('==========================================')
        print('Beginning experiment!')
        #
        print('Begin learn...')
        self.before_run()
        times = []

        for episode_num in range(1, self.episodes+1):
            self.episode_num = episode_num
            # if episode_num % 5 == 0:
            # print('reached episode', episode_num)
            self.flags.exploit = False
            self.flags.learn = True
            self.before_episode()
            print('Running {} episode {}.'.format('learn' if self.flags.learn else 'exploit', episode_num))

            start = timer()
            episode_trajectory = self.run_episode()
            end = timer()
            time_passed = end - start
            times.append(time_passed)

            episode_trajectory.cull()
            if self.save_traj:
                e_traj = episode_trajectory.get_agent_trajectories()[0]
                self.all_traj.extend([int(i) for i in e_traj.actions])
            self.episode_trajectories.append(episode_trajectory)
            self.episode_trajectories = self.episode_trajectories[-self.num_trajectories:]  # TODO improved caching
            # print('Reward: {}'.format(episode_trajectory.get_agent_total_rewards()))
            # print('Episode {} complete'.format(episode_num))
            self.after_episode()

        self.after_run()
        print('Times')
        print(times)
        return self.finalize()

    def run_episode(self, use_eval_environment=False):

        if use_eval_environment:
            self.evaluation_episode_step = 0
            self.eval_curr_trajectory = TrajectoryCollection(self.eval_episode_max_length,
                                                             self.evaluation_environment.agent_class_map,
                                                             self.evaluation_environment.agent_class_action_domains,
                                                             self.evaluation_environment.agent_class_observation_domains)

        else:
            self.episode_step = 0
            # TODO Make more concise (but note that policy groups might change)
            self.curr_trajectory: TrajectoryCollection = TrajectoryCollection(self.episode_max_length,
                                                                          self.env.agent_class_map,
                                                                          self.env.agent_class_action_domains,
                                                                          self.env.agent_class_observation_domains)
        if use_eval_environment:
            try:
                self.evaluation_environment.reset(self.flags.visualize)
            except:
                raise ValueError("Can only use evaluation environment for Multi Sampler")
        else:
            self.env.reset(self.flags.visualize)

        self.asys.reset()

        self.on_episode_start()

        if use_eval_environment:
            while self.eval_episode_active():
                self.eval_environment_step()
        else:
            while self.episode_active():
                self.step()

        if self.flags.visualize:
            self.env.visualize()

        self.asys.end_episode(self.flags.learn)
        if use_eval_environment:
            return self.eval_curr_trajectory
        else:
            return self.curr_trajectory

    #@profile
    def step(self):
        transfer_msg = self.env.pop_last_domain_transfer()
        if transfer_msg is not None:
            self.asys.transfer_domain(transfer_msg)

        observation_request = self.asys.observe_request()

        self.before_observe()
        a_obs = self.env.observe(observation_request)
        a_obs = self.on_observe(a_obs)

        if self.flags.visualize:
            self.env.visualize()

        a_act = self.asys.get_actions(a_obs, self.flags.exploit)
        a_obs, a_act = self.on_action(a_obs, a_act)

        a_rew = self.env.update(a_act)
        update_signal = self.on_update(a_obs, a_act, a_rew)
        a_obs, a_act, a_rew = update_signal

        # Reshape data via asys for asys
        self.curr_trajectory.append(self.asys.agent_ids, a_obs, a_act, a_rew, self.env.done)
        self.asys.append_pg_signals(a_obs, a_act, a_rew, self.env.done)

        # Learn step
        if self.flags.learn:
            self.asys.learn_update()
            if (isinstance(self.asys, HierarchicalSystem.HierarchicalSystem) or
                    isinstance(self.asys, PolledSampler.PolledSampler)):
                a_obs2 = self.env.observe(observation_request)
                self.asys.check_all_agent_termination(a_obs2)
                self.asys.hierarchical_update(a_obs2, self.env.done)
        else:
            if isinstance(self.asys, HierarchicalSystem.HierarchicalSystem) or \
                    isinstance(self.asys, PolledSampler.PolledSampler):
                a_obs2 = self.env.observe(observation_request)
                self.asys.check_all_agent_termination(a_obs2)

        self.after_update()
        self.episode_step += 1

    def episode_active(self):
        """
        :return: Whether or not the episode is active (that is, can proceed to the next step).
        """
        return not self.env.done and (self.episode_step < self.episode_max_length or self.episode_max_length == 0)

    # Do not touch the following lines of code.

    def before_run(self):
        pass

    def before_episode(self):
        pass

    def on_episode_start(self):
        pass

    def before_observe(self):
        pass

    def on_observe(self, agents_observation: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        return agents_observation

    def on_action(self,
                  agents_observation: Dict[Any, np.ndarray],
                  agent_action_map: Dict[Any, np.ndarray]) -> (Dict[Any, np.ndarray], Dict[Any, np.ndarray]):
        return agents_observation, agent_action_map

    def on_update(self,
                  agents_observation,
                  agent_action_map,
                  agent_rewards) -> (Dict[Any, np.ndarray], Dict[Any, np.ndarray], Dict[Any, float]):
        return agents_observation, agent_action_map, agent_rewards

    def after_update(self):
        pass

    def after_episode(self):
        pass

    def after_run(self):
        pass

    def finalize(self):
        pass

    def get_dir_name(self, current_date, save_name):
        hour = str(current_date.strftime('%H'))
        if len(hour) == 1:
            hour = '0' + hour

        minute = str(current_date.minute)
        if len(minute) == 1:
            minute = '0' + minute

        dir_name = '{}-{}-{}-{}_{}{}'.format(save_name, current_date.month, current_date.day, current_date.year, hour,
                                             minute)
        return dir_name

    def make_results_directory(self, current_date, save_name):

        # Creates a directory to save results, uses date and time for unique name
        dir_name = self.get_dir_name(current_date, save_name)

        try:
            # Creates directory to store results
            # Will need to add _number if multiple analyses are started in a single minute
            if os.path.isdir(dir_name):
                if not os.path.isdir(dir_name + '_1'):
                    dir_name = dir_name + '_1'
                elif not os.path.isdir(dir_name + '_2'):
                    dir_name = dir_name + '_2'
                elif not os.path.isdir(dir_name + '_3'):
                    dir_name = dir_name + '_3'
                elif not os.path.isdir(dir_name + '_4'):
                    dir_name = dir_name + '_4'
            os.mkdir(dir_name)

            print('Directory:', dir_name)

        except Exception as e:
            print(e)
            pass

        return dir_name


class ControllerFlags(object):

    def __init__(self):
        self.learn = True
        self.exploit = False
        self.visualize = False
