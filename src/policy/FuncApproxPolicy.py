import numpy as np

from typing import List, Any

from common import Properties
from common.vecmath import argmax_random_tiebreak
from config import Config, ConfigItemDesc, ConfigDesc
from domain import ObservationDomain, ActionDomain
from .base import Policy
from policy.function_approximator import FunctionApproximator
from policy.policy_sampler import PolicySampler


class FuncApproxPolicy(Policy):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return super().get_class_config() + [
            # TODO add default configs
            ConfigDesc(name='policy_sampler',
                       module_package='policy.policy_sampler',
                       info='Policy Sampler to use with Policy.',
                       default_configs=[]),
            ConfigDesc(name='function_approximator',
                       module_package='policy.function_approximator',
                       info='Config for Function Approximator (FA) to use with Policy.',
                       default_configs=[])
        ]
    
    @classmethod
    def properties(cls) -> Properties:
        """
        Return the compatibility properties of the class.
        This must be implemented
        """
        return Properties(use_function_approximator=True)

    def __init__(self,
                 domain_obs: ObservationDomain,
                 domain_act: ActionDomain,
                 config: Config):
        """
        Instantiate a new Policy Class (abstract)
        Base class which all Policies derive
        :param domain_obs The observation domain
        :param domain_act The action domain
        :param config The config for the experiment
        """
        super().__init__(domain_obs, domain_act, config)
        self.sampler: PolicySampler = self.config.policy_sampler.module_class(config, domain_obs, domain_act)

        # TODO figure out what to do with passing args to FA
        self.function_approximator: FunctionApproximator =\
            self.config.function_approximator.module_class(config, domain_obs, self.sampler.num_learned_parameters)

    def get_action_probs(self, states: np.ndarray):
        """
        Gets action probabilities given the current state
        """
        fa_values = self.eval(states)
        _, prob_dist = self.sampler.sample(fa_values)
        return prob_dist

    #@profile
    def get_actions(self, states: np.ndarray, use_max: bool) -> np.ndarray:
        """
        Takes observations (called states here) and return chosen action(s)
        :param states: The observations for this step
        :param use_max: Evaluate actual learned optimal policy. That is, no exploration
        """
        fa_values = self.eval(states)

        if use_max:
            assert self.domain_act.discrete

            sampler_values = self.sampler.eval(fa_values)
            # print(fa_values, states, sampler_values)
            actions = argmax_random_tiebreak(sampler_values)
            if self.domain_act.is_compound:
                raw_action = np.array([self.domain_act.extract_sub_actions(actions)])
            else:
                raw_action = np.array([actions])
        else:
            raw_action, _ = self.sampler.sample(fa_values)

        return raw_action

    def eval(self, states: np.ndarray) -> np.ndarray:
        """
        Should run the function approximator on a feature vector
        :param states
        :return Evaluation of function approximator
        """
        return self.function_approximator.eval(states)

    def update_with_dataset(self, data: np.array):
        """Defer call to however function approximator/table decides to update itself"""
        self.function_approximator.update(data)

    def make_feature_vectors(self, states: np.ndarray) -> np.ndarray:
        """Performs policy-specific transformations on observed features if applicable."""
        return states

    def serialize(self):
        """Return format that can be saved on disk."""
        return self.function_approximator.get_variable_vals()

    def load(self, vals: Any):
        """Load from serialized format"""
        self.function_approximator.set_variable_vals(vals)

    def compile(self, loss_func, learning_rate: float):
        self.function_approximator.compile(loss_func, learning_rate)
