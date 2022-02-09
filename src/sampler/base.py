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
from agentsystem.util import PolicyGroup
from agentsystem import AgentSystem

# Divider for weight dict keys for saving policies
DIVIDER = '\x1D'


class Sampler(AgentSystem, abc.ABC):

    def __init__(self,
                 agent_id_class_map: Dict[int, int],
                 agent_class_action_domains: Dict[int, ActionDomain],
                 agent_class_observation_domains: Dict[int, ObservationDomain],
                 auxiliary_data: Dict[str, Any],
                 config: Config):
        """
        Initializes a sampler, which is a Agent System made for collect samplers for offline learning
        Was not made with the intention of acting as a standard Agent System for online learning

        :param agent_id_class_map: Map of agent IDs to agent class IDs.
        :param agent_class_action_domains: Map of agent class IDs to the associated ActionDomains
        :param agent_class_observation_domains: Map of agent class IDs to the associated ObservationDomains
        :param auxiliary_data: Other information passed in via a string-keyed dictionary
        :param config: ExperimentConfig for experiment
        """
        super.__init__(agent_id_class_map,
                 agent_class_action_domains,
                 agent_class_observation_domains,
                 auxiliary_data,
                 config)


