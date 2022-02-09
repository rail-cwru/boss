"""
Base class for Environment
"""
import abc
import numpy as np
from typing import List, Dict, Tuple, TYPE_CHECKING, Union, Any

from common.aux_env_info import AuxiliaryEnvInfo
from common.domain_transfer import DomainTransferMessage
from domain.observation import ObservationDomain
from domain.ActionDomain import ActionDomain
from config.moduleframe import AbstractModuleFrame

if TYPE_CHECKING:
    from config import Config


class Environment(AbstractModuleFrame):

    def __init__(self, config: 'Config'):
        """
        Initialize an Environment and all fields. Also check compatibility with  other configs.

        The following should be initialized here or in self._reset_state().
            self._agent_id_list
            self._agent_class_id_map
            self._agent_class_list
            self.agent_class

        :param config: Config object containing configuration and compatbility information for all aspects of the
            experiment
        """
        self.set_initial_seed(config.seed)

        # The config for the whole experiment TODO (EVENTUALLY SHOULDN'T HAVE)
        self.all_config = config

        # Its own environment config
        self.config = config.find_config_for_instance(self)
        self.state: Any = []
        self.action_domain: List[ActionDomain] = self._create_action_domains(config)
        # Since Observation Domains could depend on action domains (e.g. if they are ActionFeatureDomain)
        # We should initialize ActionDomains first.
        self.agent_class_action_domains: Dict[Any, ActionDomain] =\
            self._create_action_domains(config)
        self.agent_class_observation_domains: Dict[Any, ObservationDomain] =\
            self._create_observation_domains(config)
        # TODO make completely optional
        self.last_domain_transfer: DomainTransferMessage = None
        self.gather_metrics = False
        self.metrics: List[Any] = []
        self.done: bool = False
        self.reward_range = 10

    def get_auxiliary_info(self) -> AuxiliaryEnvInfo:
        # TODO somehow associate with properties / allow other modules to know what's in here
        return AuxiliaryEnvInfo()

    def get_state(self) -> np.ndarray:
        """
        Get the entire current state.
        :return: EnvironmentState
        """
        return self.state

    def reset(self, visualize: bool = False):
        self.state = self._reset_state(visualize)
        # DOMAIN TRANSFER MIGHT HAVE TAKEN PLACE RECENTLY
        self.done = False
        self.metrics = []

    def pop_last_domain_transfer(self) -> DomainTransferMessage:
        """
        Pop the last domain transfer received by the Environment.

        The attribute env.last_domain_transfer should be updated when:
            1. env.transfer_domain is called
            2. The domain is altered in env.update
        :return: The last domain transfer message.
        """
        ret = self.last_domain_transfer
        self.last_domain_transfer = None
        return ret

    @abc.abstractmethod
    def set_initial_seed(self, seed: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_seed_state(self, seed_state):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_seed_state(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, actions: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Update the current state.

        If the domain changes (e.g if the update removes an agent), still return reward for that agent this time.
        However, the ``self.last_domain_transfer`` should be updated to reflect the most recent change.

        :param actions: ndarray specifying actions for each agent in state.
        :return: rewards for each agent.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def observe(self, obs_groups : List[Tuple[int,...]] = None) -> Dict[Union[int, Tuple[int, ...]], np.ndarray]:
        """
        Get the observations for the current state
        :return: Observation for each agent.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _reset_state(self) -> Any:
        """
        Initialize the environment features to their initial state using the config object.
        :return: Arbitrary object describing the environment's state.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_observation_domains(self, config) -> Dict[Union[str,int], ObservationDomain]:
        """
        Initialize the observation domain object for this Environment.
        :param config: Config object for experiment
        :return: List[ObservationDomain] List of observation domains for each agent.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def agent_class_map(self) -> Dict[int, int]:
        """returns mapping from canonical agentId to agent class."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def agent_class_list(self) -> List[str]:
        """returns a list of possible agent classes"""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def agent_id_list(self) -> List[int]:
        """
        Returns a list of canonical agent ids.

        The list of agent IDs must always be a contiguous, ascending list of integers starting with 0.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_action_domains(self, config) ->Dict[Union[str,int], ActionDomain]:
        """
        Set the observation domain and action domain for this environment
        :param config: Config object
        :return: pair of observation domain and action domain.
        """
        raise NotImplementedError()

    def transfer_domain(self, message: DomainTransferMessage) -> DomainTransferMessage:
        """
        Add or delete agents.
        :param message: Domain transfer to perform on environment.
        :return: DomainTransferMessage (for agentsystem) which includes agent-classes also.
        """
        raise NotImplementedError()

    def visualize(self):
        """
        Optionally visualize the environment.
        """
        pass

    def set_gather_metrics(self, gather_metrics: bool):
        """
        Request special metrics. Might exist for some environments.

        Typically special metrics might be some other sort of pertinent data which only the environment knows
        whether or not to keep track of.

        Request metrics before an episode is run. This allows us to avoid gathering metrics when we do not need to
        in order to save computation.
        """
        self.gather_metrics = gather_metrics

    def get_metrics(self):
        """
        :return: Gathered metrics.
        """
        # TODO make this more compatible with property checks, etc
        assert self.gather_metrics
        return self.metrics


class HandcraftEnvironment(Environment):
    """
    An environment which has inbuilt,handcrafted actions, usually meant as a "perfect baseline."
    """

    # TODO prop HANDCRAFTED_ACTIONS=True while False for Environment (default)?

    @abc.abstractmethod
    def handcraft_actions(self) -> Dict[int, np.ndarray]:
        pass
