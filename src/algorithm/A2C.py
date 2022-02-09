from typing import List
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.distributions import Categorical
import abc
import numpy as np

from common.properties import Properties
from config import Config, ConfigItemDesc, checks
from common.trajectory import Trajectory
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA
from policy.function_approximator.ACFunctionApproximator import ACFunctionApproximator
from . import Algorithm
from policy import Policy
from model import Model
from .util import MemoryReplay, calc_discounted_rewards

class A2C(Algorithm):
    """
    https://arxiv.org/pdf/1602.01783.pdf
    """

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return Algorithm.get_class_config() + [
            ConfigItemDesc(name='learning_rate', check=checks.unit_float, info='Actor learning rate. A decimal in [0, 1]'),
            ConfigItemDesc(name='entropy_coeff', check=checks.unit_float, info='Entropy coefficient within loss function. A decimal in [0, 1]'),
            ConfigItemDesc(name='value_coeff', check=checks.unit_float, info='Value coefficient within loss function. A decimal in [0, 1]')
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
        self.entropy_coeff = self.config.entropy_coeff
        self.value_coeff = self.config.value_coeff
        self.memory = MemoryReplay(self.config.memory_size, self.config.batch_size, False, self.config.update_interval)

    def compile_policy(self, policy: Policy):
        pytorch_mse= MSELoss()

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
                discounted_rewards = calc_discounted_rewards(self.discount_factor, rewards, False)

                if i == 0:
                    batch_states = states
                    batch_actions = actions
                    batch_discounted_rewards = discounted_rewards
                else:
                    batch_states = np.concatenate((batch_states, states), axis=0)
                    batch_actions = np.concatenate((batch_actions, actions), axis=0)
                    batch_discounted_rewards = np.concatenate((batch_discounted_rewards, discounted_rewards), axis=0)

            # Convert shape
            batch_discounted_rewards = np.reshape(batch_discounted_rewards, (batch_discounted_rewards.shape[0], 1))

            # Convert to tensors
            x_var = torch.tensor(batch_states, dtype=torch.float, device=policy.function_approximator.device)
            a_var = torch.tensor(batch_actions, dtype=torch.long, device=policy.function_approximator.device)
            dr_var = torch.tensor(batch_discounted_rewards, dtype=torch.float, device=policy.function_approximator.device)

            # Predict values
            y_pred = policy.function_approximator.eval_critic_tensor(x_var)
            advantage = torch.sub(dr_var, y_pred)

            # Calculate value loss
            value_loss = pytorch_mse(y_pred, dr_var)

            # Predict probability from policy
            fa_values = policy.function_approximator.eval_tensor(x_var)

            # Get probabilities from sampler
            prob = policy.sampler.sample_tensor(fa_values)

            # Calculate entropy
            cat_dist = Categorical(prob)
            entropy = cat_dist.entropy().mean()

            # Downselect -log probability given the action
            selected_log_probs = -cat_dist.log_prob(a_var)

            # Finish policy loss calc
            weighted_advantage = torch.mul(selected_log_probs, advantage)
            policy_loss = torch.sum(weighted_advantage)

            loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
            return loss

        fa = policy.function_approximator
        assert isinstance(fa, ACFunctionApproximator)

        policy.compile(pytorch_loss, self.learning_rate)

    def update(self, policy: Policy, model: Model, trajectory: Trajectory):
        # Actor update is the same as REINFORCE, which is a offline algorithm, if trajectory is not complete do not update
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
