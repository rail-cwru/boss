""" Certain data appeared too many times.
If we can regulate its appearance and make more sensible its structure, we should encapsulate and bring over here.
Slightly unfortunate that we ended up with this... preferably, have synchronized references in each object, but...

Prefer Cython ASAP in the future.
"""

from typing import Dict, List, Union, Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from domain.observation import ObservationDomain
    from domain.ActionDomain import ActionDomain


class TrajectoryCollection(object):
    """
    Data class for managing trajectories for whole episodes:
        A map from agent keys to a time-ordered Trajectory object handling [Observation, Action, Reward] data.

    Contains both trajectories organized by agents and trajectories organized by policy groups
    in order to avoid repeated accumulation and construction of larger arrays between environment(s) and AgentSystem.

    Should be stored in Controller.
    """

    def __init__(self,
                 max_time: int,
                 agent_id_class_map: Dict[int, int],
                 agent_class_action_domains: Dict[int, 'ActionDomain'],
                 agent_class_observation_domains: Dict[int, 'ObservationDomain']):

        self.__agent_data: Dict[int, Trajectory] = {}
        for a1, ac1 in agent_id_class_map.items():
            obs_dom = agent_class_observation_domains[ac1]
            act_dom = agent_class_action_domains[ac1]
            self.__agent_data[a1] = Trajectory.allocate(max_time=max_time, obs_domain=obs_dom, act_domain=act_dom)

        self.time = 0
        self.max_time = max_time

    def __len__(self):
        return self.time

    def append(self,
               agent_ids: List[int],
               observations: Dict[int, np.ndarray],
               actions: Dict[int, np.ndarray],
               rewards: Dict[int, Union[int, float]],
               done=False):
        """
        Append a datapoint for a single agent to the trajectory.
        :param agent_ids: agent ids
        :param observations: agent observation map
        :param actions: agent action map
        :param rewards: agent reward map
        :param done: If the observed state is terminal
        """
        for agent_id in agent_ids:
            # TODO somehow we should be able to get the obs/act domains of new agents in case we need to add them
            self.__agent_data[agent_id].append(observations[agent_id], actions[agent_id], rewards[agent_id])
            self.__agent_data[agent_id].set_done(done)
        self.time += 1

    def get_agent_trajectories(self):
        """
        Default behavior is to return the agent_id organized trajectories.
        Call sparingly.
        :return: Trajectories organized by agent ID
        """
        return self.__agent_data

    def get_agent_trajectory(self, agent_id):
        return self.__agent_data[agent_id]

    def get_agent_total_rewards(self) -> Dict[Any, float]:
        return {agent: np.sum(a_trajectory.rewards) for agent, a_trajectory in self.__agent_data.items()}

    def cull(self):
        """
        Cull all data arrays. Use with caution. Small arrays are not deallocated well and may cause memory leaks.
        """
        for trajectory in self.__agent_data.values():
            trajectory.cull()


class Trajectory(object):
    """
    A time-ordered data object that efficiently manages access to Observations, Actions, Rewards, and maintains
        a rolling cumulative reward.
    """

    @classmethod
    def allocate(cls, max_time: int, obs_domain: 'ObservationDomain', act_domain: 'ActionDomain'):
        """
        Allocate the trajectory data given the observation and action domain.

        Since reward is consistently a single scalar, we do not need that information.
        :param max_time: The maximum length of the episode the trajectory is recorded from.
        :param obs_domain: Observation domain
        :param act_domain: Action domain
        """
        observations = np.zeros((max_time,) + obs_domain.shape, obs_domain.widest_dtype)
        if act_domain.is_compound:
            act_domain_shape = (len(act_domain.sub_actions),)
        else:
            act_domain_shape = act_domain.shape
        actions = np.zeros((max_time,) + act_domain_shape, act_domain.widest_dtype)
        rewards = np.zeros(max_time, 'float32')
        return cls(observations, actions, rewards, time=0)

    def __init__(self, obs, act, rew, time, done=False):

        self.__observations = obs
        self.__actions = act
        self.__rewards = rew
        self.time = time
        self.done = done

    def append(self, obs, act, reward):
        """
        Append data to the trajectory.
        :param obs: Observation data
        :param act: Action data
        :param reward: Reward data
        """
        self.__observations[self.time] = obs
        self.__actions[self.time] = act
        self.__rewards[self.time] = reward
        self.time += 1

    @property
    def observations(self) -> np.ndarray:
        return self.__observations[:self.time]

    @property
    def actions(self) -> np.ndarray:
        return self.__actions[:self.time]

    @property
    def rewards(self) -> np.ndarray:
        return self.__rewards[:self.time]

    def cull(self):
        """
        Trim unused array space in case of early episode termination.
        """
        # Use np.copy to destroy reference to old, letting numpy deallocate memory.
        self.__observations = np.copy(self.__observations[:self.time])
        self.__actions = np.copy(self.__actions[:self.time])
        self.__rewards = np.copy(self.__rewards[:self.time])

    def __getitem__(self, slicer):
        return Trajectory(self.observations[slicer], self.actions[slicer], self.rewards[slicer],
                          time=self.time,
                          done=self.done)

    def __len__(self):
        # TODO deal with extracted Trajectory partitions having unreliable length
        return min(len(self.observations), self.time)

    def set_done(self, done):
        self.done = done
