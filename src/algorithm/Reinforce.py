from typing import List
import torch
from torch.nn import CrossEntropyLoss
import abc
import numpy as np
from scipy.special import logsumexp

from common.properties import Properties
from config import Config, ConfigItemDesc, checks
from common.trajectory import Trajectory
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA
from policy.function_approximator.ACFunctionApproximator import ACFunctionApproximator
from . import Algorithm
from policy import Policy
from model import Model
from .util import MemoryReplay, calc_discounted_rewards



class Reinforce(Algorithm):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return Algorithm.get_class_config() + [
            ConfigItemDesc(name='learning_rate', check=checks.unit_float, info='Learning rate. A decimal in [0, 1]'),
        ]

    @classmethod
    def properties_helper(cls):
        return Properties(
            use_function_approximator=True,
            pytorch=True)

    def __init__(self, config: Config):
        """
        TODO writeme
        """
        Algorithm.__init__(self, config)
        self.learning_rate = self.config.learning_rate
        self.memory = MemoryReplay(self.config.memory_size, self.config.batch_size, False, self.config.update_interval)

    def compile_policy(self, policy: Policy):
        pytorch_cross_entropy = CrossEntropyLoss(reduction='none')

        def numpy_cross_entropy(probs, actions):
            lse = logsumexp(probs, axis=1)
            ce = -probs[range(actions.shape[0]), actions] + lse
            return ce

        def numpy_loss(data: np.array):
            raise Exception('Numpy version of Reinforce is not complete.')

            # Gather states, actions and discounted rewards
            num_episodes = len(data)
            batch_states = np.array([])
            batch_actions = np.array([])
            batch_discounted_rewards = np.array([])
            for i in range(num_episodes):
                states, actions, rewards, _, _, _ = data[i]

                batch_states = np.concatenate((batch_states, states), axis=0)
                batch_actions = np.concatenate((batch_actions, actions), axis=0)
                discounted_rewards = calc_discounted_rewards(self.discount_factor, rewards)
                batch_discounted_rewards = np.concatenate((batch_discounted_rewards, discounted_rewards), axis=0)

            # Get function approximator values
            fa_values = policy.function_approximator.eval(batch_states)

            # Get probabilities from sampler
            probs = np.zeros(fa_values.shape)
            for i in range(fa_values.shape[0]):
                _, probs[i,:] = policy.sampler.raw_sample(fa_values[i,:])

            # Use cross entropy to control numerical errors
            selected_log_probs = numpy_cross_entropy(probs, batch_actions)

            # Finish loss calc
            advantage = selected_log_probs * batch_discounted_rewards
            sum_advantage = np.sum(advantage)
            return sum_advantage

        def pytorch_loss(data: np.array):
            # Gather states, actions and discounted rewards
            num_episodes = len(data)
            batch_states = None
            batch_actions = None
            batch_discounted_rewards = None
            for i in range(num_episodes):
                states, actions, rewards, _, _, _ = data[i]

                if policy.domain_act.is_compound:
                    actions = self.translate_compound_action(policy.domain_act, actions)
                actions = np.reshape(actions, (actions.shape[0]))
                discounted_rewards = calc_discounted_rewards(self.discount_factor, rewards, True)

                if i == 0:
                    batch_states = states
                    batch_actions = actions
                    batch_discounted_rewards = discounted_rewards
                else:
                    batch_states = np.concatenate((batch_states, states), axis=0)
                    batch_actions = np.concatenate((batch_actions, actions), axis=0)
                    batch_discounted_rewards = np.concatenate((batch_discounted_rewards, discounted_rewards), axis=0)

            # Convert to tensors
            x_var = torch.tensor(batch_states, dtype=torch.float, device=policy.function_approximator.device)
            a_var = torch.tensor(batch_actions, dtype=torch.long, device=policy.function_approximator.device)
            dr_var = torch.tensor(batch_discounted_rewards, dtype=torch.float, device=policy.function_approximator.device)

            # Predict probability from policy
            fa_values = policy.function_approximator.eval_tensor(x_var)

            # Get probabilities from sampler
            prob = policy.sampler.sample_tensor(fa_values)

            # Use cross entropy to control numerical errors
            selected_log_probs = pytorch_cross_entropy(prob, a_var)

            # Finish loss calc
            advantage = torch.mul(selected_log_probs, dr_var)
            sum_advantage = torch.sum(advantage)
            return sum_advantage

        # Check to make sure not actor critic
        fa = policy.function_approximator
        assert not isinstance(fa, ACFunctionApproximator)

        # Compile policy
        if isinstance(fa, AbstractPyTorchFA):
            policy.compile(pytorch_loss, self.learning_rate)
        else:
            policy.compile(numpy_loss, self.learning_rate)

    def update(self, policy: Policy, model: Model, trajectory: Trajectory):
        # REINFORCE is a offline algorithm, if trajectory is not complete do not update
        if not trajectory.done:
            return 

        # Add to memory
        self.memory.log(trajectory)

        # Check if update interval has been met
        if not self.memory.update:
            return

        # Sample memory
        batch = self.memory.sample()
        policy.update_with_dataset(batch)
