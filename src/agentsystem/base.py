"""
Base classes for AgentSystem.
"""

import abc
from typing import List, Dict, Tuple, Union, Optional, Type, Any
import numpy as np

from policy.function_approximator.pytorch_fa import AbstractPyTorchFA
from common.domain_transfer import DomainTransferMessage
from domain import ObservationDomain, ActionDomain
from config import Config
from algorithm import Algorithm
from policy import Policy
from .util import PolicyGroup

# Divider for weight dict keys for saving policies
DIVIDER = '\x1D'


class AgentSystem(abc.ABC):

    def __init__(self,
                 agent_id_class_map: Dict[int, int],
                 agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain],
                 auxiliary_data: Dict[str, Any],
                 config: Config):
        """
        Initialize an AgentSystem (abstract). An Agentsystem is a group of potentially interacting agents.

        Imeiro defines an AgentSystem in such a way that learners (PolicyGroup) does not need to have a 1-to-1 map to
            agents. In fact, the mapping can be many-to-many. The AgentSystem will define how the agents map to
            learners, how learning (and coordination, if present) is conducted, and how actions are extracted
            over the whole of the system.

        An AgentSystem typically should include a Policy and an Algorithm. The Algorithm will be used to update
            the policy.

        (TODO) this paper was found after creating this framework. It is unsure currently how our system compares.
        [1] N. Vlassis. A concise introduction to multiagent systems and distributed AI. Informatics Institute,
                        University of Amsterdam, Sept. 2003. http://www.science.uva.nl/˜vlassis/cimasdai (?)

        AgentSystem is an Abstract Class which manages agents in an environment,
            keeping control over the policy or policies for the agents,
            and determining how to use an algorithm to update the policies for those agents.
            AgentSystem will encapsulate all coordination behavior if it exists and provide a neat interface
            from which the actions of the agents can be requested, given observations from an environment.

        AgentSystem should receive the observation domains and action domains from the Environment.
            These are descriptors of the observed features for different agents, and the possible actions for
            different agents as well.

        In order to support both single-agent and multi-agent structures,
            AgentSystem maintains a map of agents to policy groups, which are collections of agents that share a policy
            and also a map of policy groups to the actual policy objects,
            in order to facilitate policy sharing among groups.

        For example, if there are multiple groups of two which all use the same policy,
            then the design of using both these maps makes sense in that the same two-agent policy would be
            shared for all the two-agent policy groups.

        :param agent_id_class_map: Map of agent IDs to agent class IDs.
        :param agent_class_action_domains: Map of agent class IDs to the associated ActionDomains
        :param agent_class_observation_domains: Map of agent class IDs to the associated ObservationDomains
        :param auxiliary_data: Other information passed in via a string-keyed dictionary
        :param config: ExperimentConfig for experiment

        Examples of auxiliary data:
            optional joint_agent_class_observation_domains:
                Map of tuples of agent class IDs to the joint observation domains
        """
        self.top_config = config

        # TODO put policy and algorithm into asys requirements FOR SUBCLASSES
        # TODO figure out how on earth to get it organized sanely
        self.policy_class: Type[Policy] = self.top_config.policy.module_class

        # TODO - remove all references, use instantiated algorithm where possible
        # ASys owns algorithm .... should only IndepSys and SharedSys have the single alg?
        # Where along the ladder do heterogeneous algorithms / policies exist?
        # TODO heterogeneous alg only by modification?
        self.algorithm = self.top_config.algorithm.module_class(config)
        self.algorithm_class: Type[Algorithm] = self.top_config.algorithm.module_class

        # Map from CANONICAL AGENT ID (uuid) to AGENT CLASS ID
        self.agent_id_class_map: Dict[int, int] = agent_id_class_map

        # CANONICAL AGENT IDs (uuid)
        self.agent_ids: List[int] = [key for key in agent_id_class_map.keys()]
        self.agent_classes: List[int] = list(set(agent_id_class_map.values()))
        sorted(self.agent_classes)

        # Inverse map from agent class ID to list of agents
        self.agent_class_id_map: Dict[int, List[int]] = {agent_class: [agent_id for agent_id, agent_id_class
                                                                       in self.agent_id_class_map.items()
                                                                       if agent_id_class == agent_class]
                                                         for agent_class in self.agent_classes}

        # Action Domain for each Agent Class
        self.agent_class_action_domains: Dict[int, ActionDomain] = agent_class_action_domains

        # Observation Domain for each Agent Class
        self.agent_class_observation_domains: Dict[int, ObservationDomain] = agent_class_observation_domains

        # Method of production determined by concrete class
        self.max_time = config.episode_max_length  # TODO find? remove if annoying? etc.
        self.policy_groups: List[PolicyGroup] = self._make_policy_groups(
            self.agent_id_class_map,
            self.agent_class_action_domains,
            self.agent_class_observation_domains)

        # agent_id -> list[policy_groups]
        self.agent_policy_groups_map: Dict[int, List[PolicyGroup]] = {agent_id: [] for agent_id in self.agent_ids}
        for pg in self.policy_groups:
            for agent_id in pg.agents:
                self.agent_policy_groups_map[agent_id].append(pg)

        # Hook in all algorithms and policies
        for pg in self.policy_groups:
            self.algorithm.compile_policy(pg.policy)

        # Validate observation request output
        observation_request = self.observe_request()
        if observation_request is not None:
            for obs_key in observation_request:
                msg_err = 'Encountered an invalid agent or joint agent of the form [{}]' \
                          'for which an observation was requested.'.format(obs_key)
                assert isinstance(obs_key, int) or (isinstance(obs_key, tuple) and len(obs_key) > 1), msg_err

    @abc.abstractmethod
    def _make_policy_groups(self,
                            agent_class_map: Dict[int, int],
                            agent_class_action_domains: Dict[int, ActionDomain],
                            agent_class_observation_domains: Dict[int, ObservationDomain]) -> List[PolicyGroup]:
        """
        AgentSystem concrete subclass-specific realization of how to partition agents into policy groups.
        We pass in action domains and observation domains since it may make use of domain information in partition,
            if the algorithm is conditioned on properties belonging to those.

        We determine in this function how to also assign Policies and Models to the PolicyGroups.
            Each PolicyGroup should be associated with exactly one Policy and Model (dependent on the Algorithm used).
            The PolicyGroup will instantiate a trajectory object.

        Currently, there should not be any ASys which should implement policy groups that cross agent classes.
        :param agent_class_action_domains: Agent class action domains.
        :param agent_class_observation_domains: Agent class observation domains.
        :return: List of Policy Group objects.
        """
        raise NotImplementedError()

    def observe_request(self) -> Optional[List[Tuple[int]]]:
        """
        A special function which is used for requesting joint features from the Environment.

        If the environment property enables use_joint_features, observe_request can return a list of tuples.
            Each tuple in the list is a tuple of AgentIDs for which a single joint observation should be created.
            For example, for the agents from agent_class_map [0, 1, 2, 3, 4, 5]
        The observe_request [(0, 1), (2, 3), (3, 4), (3, 5), (5,)]
            would indicate that we want joint features for the first four pairs,
            and the single feature for the fifth, the agent 5.

        By default, observe_request returns None.
            Under this behavior, the controller will bypass the observe_request step.
            No joint features will be requested from the environment.

        :return: None if there are no joint features, otherwise a list of tuples.
        """
        return None

    def agent_keys(self) -> List[Union[int, Tuple[int, ...]]]:
        """
        Return keys of agent-centric observations for use with trajectory.
        :return: List of ints and/or tuples. If observe_request is None then it should just be agent_ids.
        """
        request = self.observe_request()
        if request is None:
            return self.agent_ids
        else:
            return request

    @abc.abstractmethod
    def translate_pg_signal(self,
                            a_observations: Dict[int, np.ndarray],
                            a_actions: Dict[int, np.ndarray],
                            a_rewards: Dict[int, Union[int, np.ndarray]],
                            ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, Union[int, np.ndarray]]]:
        """
        Translate the (observation / action / reward) signal from agents to policygroups.

        In default cases, the agent mapping will be the same as the pg mapping.
        Otherwise, where the policy groups do not directly represent the agent, this function will be nontrivial.
        Usually such cases would be since PG represents some sort of joint agent or meta-agent.

        Some AgentSystems might use cached information to avoid recomputing already-joint actions, rewards, etc.

        To prevent cases where this independent case is mistakenly inherited by an asys which uses joint agents,
        "meta-agents," we require this method to be overridden.

        All inputs are optional; it is expected that if an input is ignored, the corresponding output will be ignored.

        :param a_observations: Agent-organized observation map
        :param a_rewards: Agent-organized reward map
        :param a_actions: Agent-organized action map
        :return: Policy Group-organized observation / reward / action map as tuple
        """
        raise NotImplementedError()

    def append_pg_signals(self,
                          a_obs: Dict[int, np.ndarray],
                          a_act: Dict[int, np.ndarray],
                          a_rew: Dict[int, Union[int, np.ndarray]],
                          done: bool):
        """
        Append a datapoint for a policy group (n x m arrays) to the trajectory.
        :param a_obs: Agent-organized observation map
        :param a_act: Agent-organized action map
        :param a_rew: Agent-organized reward map
        :param done: If the observed state is terminal
        """
        # Possibly fuse entire function into translate pg signal if it's a significant performance hit
        pg_obs, pg_act, pg_rew = self.translate_pg_signal(a_obs, a_act, a_rew)

        # TODO Figure out how to distinguish btw vanished PGs and PGs with updates on hold
        # TODO (delete when confirmed safe) Used to delete pgs that weren't updated...!?
        # THIS WILL NOT WORK SAFELY FOR POMDPs. (that means you too, HRL)

        # TODO (caution?) used to alloc new traj if pg id didn't exist in list...
        # Now that traj is allocated with pg, it shouldn't be a problem
        # Delete this comment when it's confirmed that it's no longer a problem

        for policy_group in self.policy_groups:
            policy_group.append(pg_obs, pg_act, pg_rew, done)

    @abc.abstractmethod
    def learn_update(self):
        """
        Apply a learning step to the Agent System with the specified algorithm
        and the agent class trajectories for the episode, which consist of a list of
        tuples of observations, actions, and reward.

        Internally uses a list of trajectories:
            The trajectories being a list of tuples of
                observations, actions, rewards.
        The dimensions of the ndarrays are [a * c], where:
            a is the number of agents in that agentclass
            c is the number of features in that data

        The method should be implemented appropriately for the concrete AgentSystem class.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_actions(self, observations: Dict[int, np.ndarray], use_max: bool = False) -> Dict[int, np.ndarray]:
        """
        Return the map of agent class to actions when the system receives observations.
        :param observations: Map of canonical agent ID to observations.
        :param use_max: Exploit (if meaningful for the system).
        :return: Map of canonical agent ID to actions.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer_domain(self, domain_transfer_method: DomainTransferMessage) -> 'AgentSystem':
        """
        Transfer from the old AgentSystem to a new agentsystem with new domains.

        AgentSystems which implement many choices of domain transfer may have this method configurable.

        Compatibility must be checked with policy and transfer method.

        Transfer Domain only allows transferring domain within the scope of the DomainTransfer M
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset episode-tied information when starting a new episode.
        Could potentially be overwritten.
        """
        for pg in self.policy_groups:
            pg.allocate_new_trajectory()

    def end_episode(self, learn: bool = False):
        """
        Called when ending the episode.

        This delegates the end_episode function to the algorithm to call it on each policy group.
        :return:
        """
        # self.i += 1
        # self.means.append(np.mean(self.epi_means))
        # self.epi_means = []
        # #if self.i == 100:
        # print(self.means)

        for pg in self.policy_groups:
            if hasattr(pg.policy, 'sampler') and learn:
                pg.policy.sampler.update_params()
            self.algorithm_class.end_episode(pg.policy, pg.model)
            pg.trajectory.cull()

    def save(self, save_file: str):
        """
        Save the learned agentsystem to a file.
        :param save_file: File location.
        """
        # TODO in the future, use a better saving method, preferably delegated to policy / FA.
        save_dict = self.get_save_dict()
        np.savez_compressed(save_file, **save_dict)

    def get_save_dict(self):
        """
        Get the save data without saving to a file.
        """
        save_dict = {}
        for pg in self.policy_groups:
            escaped_prefix = '{}{}'.format(pg.pg_id, DIVIDER)
            policy: Policy = pg.policy
            if isinstance(policy, Policy):
                weight_dict = policy.serialize()
                for weight_name, weight_arr in weight_dict.items():
                    full_name = escaped_prefix + weight_name
                    save_dict[full_name] = weight_arr
            else:
                print('Policy of class [{}] was not saved because it had nothing to save.'.format(type(policy)))
        return save_dict

    def load(self, load_file: str):
        """
        Load the learned policies from a file.
        :param load_file: File location.
        """
        # TODO in the future, use a better loading method, preferably delegated to policy / FA.
        # TODO assert (somehow) that the saved data is compatible with this instance
        npz = np.load(load_file)
        pg_weight_dicts = {}
        waiting_load = []
        for full_name in npz.files:
            pg_id_str, weight_name = full_name.split(DIVIDER)
            pg_id = int(pg_id_str)
            weight_arr = npz[full_name]
            if pg_id not in pg_weight_dicts:
                pg_weight_dicts[pg_id] = {}
            pg_weight_dicts[pg_id][weight_name] = weight_arr
            waiting_load.append(pg_id)
        for pg_id, weight_dict in pg_weight_dicts.items():
            for pg in self.policy_groups:
                if pg.pg_id == pg_id:
                    try:
                        policy: Policy = pg.policy
                        policy.load(weight_dict)
                        waiting_load.remove(pg_id)
                    except Exception as e:
                        print('Encountered an error loading policies from [{}]'.format(load_file))
                        raise e
        for nonloaded in waiting_load:
            print('WARNING: Policy for policy group ID [{}] was not loaded from [{}]'.format(nonloaded, load_file))

    def _get_agent_observation_domain(self, agent_id) -> ObservationDomain:
        """
        Helper method to obtain agent's observation domain using the agent class map.
        """
        return self.agent_class_observation_domains[self.agent_id_class_map[agent_id]]

    def _get_agent_action_domain(self, agent_id) -> ActionDomain:
        """
        Helper method to obtain agent's action domain using the agent class map.
        """
        return self.agent_class_action_domains[self.agent_id_class_map[agent_id]]
