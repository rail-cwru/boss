import abc
import torch
from torch.nn import MSELoss
from typing import List, TYPE_CHECKING
import numpy as np

from algorithm import Algorithm
from common import Properties, Trajectory
from config import Config, checks, ConfigItemDesc
from model import Model
from policy import Policy
from policy.function_approximator.pytorch_fa import AbstractPyTorchFA
from policy.function_approximator.ACFunctionApproximator import ACFunctionApproximator
from policy import TabularPolicy
from .util.memory_replay import MemoryReplay

class TemporalDifference(Algorithm):

    @classmethod
    def get_class_config(cls) -> List[ConfigItemDesc]:
        return Algorithm.get_class_config() + [
            ConfigItemDesc(name='learning_rate', check=checks.unit_float, info='Learning rate. A decimal in [0, 1]'),
            ConfigItemDesc(name='is_online', check=checks.boolean, info='Boolean indicating whether to use online learning.'),
        ]

    @classmethod
    def properties_helper(cls):
        return Properties(use_discrete_action_space=True)

    def __init__(self, config: Config):
        """
        TODO writeme
        """
        Algorithm.__init__(self, config)
        self.learning_rate = self.config.learning_rate
        self.online = self.config.is_online
        self.memory = MemoryReplay(self.config.memory_size, self.config.batch_size, self.online, self.config.update_interval)

    def compile_policy(self, policy: Policy):
        def calculate_target(policy, state, action, reward, done, next_state, next_action):
            actions_values = policy.eval(state).flatten()
            value = actions_values[action]

            if done:
                # If tabular need to calculate gradient here
                if isinstance(policy, TabularPolicy):
                    target = reward - value
                else:
                    target = reward 
            else:
                next_action_value = self._get_action_value(policy, next_state, next_action)
                td_target = reward + self.discount_factor * next_action_value

                if isinstance(policy, TabularPolicy):
                    target = td_target - value
                else:
                    target = td_target 
            return target

        def preprocess(policy, data):
            # Gather states, actions and discounted rewards
            batch_size = len(data)
            batch_states = None
            batch_actions = None
            batch_targets = None
            for i in range(batch_size):
                if not self.online:
                    state, action, reward, done, next_state, next_action = data[i]

                    # Translate complex actions
                    if policy.domain_act.is_compound:
                        action = self.translate_compound_action(policy.domain_act, action)
                    action = np.reshape(action, (action.shape[0]))

                    episode_length = len(state)
                    target = np.zeros((reward.shape[0]))
                    for j in range(episode_length):
                        target[j] = calculate_target(policy, state[j], action[j], reward[j], done[j], next_state[j], next_action[j])
                else:
                    state, action, reward, done, next_state, next_action = data[i]

                    # Translate complex actions
                    if policy.domain_act.is_compound:
                        action = policy.domain_act.make_compound_action(action)
                    else:
                        action = action[0]

                    target = calculate_target(policy, state, action, reward, done, next_state, next_action)

                    # Convert to numpy array for similarity
                    state = np.array([state])
                    action = np.array([action])
                    target = np.array([target])

                if i == 0:
                    batch_states = state
                    batch_actions = action
                    batch_targets = target
                else:
                    batch_states = np.concatenate((batch_states, state), axis=0)
                    batch_actions = np.concatenate((batch_actions, action), axis=0)
                    batch_targets = np.concatenate((batch_targets, target), axis=0)

            return batch_states, batch_actions, batch_targets

        def numpy_mse(y_pred: np.ndarray, y_var: np.ndarray):
            return 0.5 * np.square(np.subtract(y_pred, y_var)).mean()

        pytorch_mse = MSELoss()

        def numpy_loss(data):
            raise Exception('Numpy version of TemporalDifference is not complete.')

            batch_states, batch_actions, batch_targets = preprocess(policy, data)
            batch_size = batch_actions.shape[0]

            # Get function approximator values
            predicted_q_vals = policy.function_approximator.eval(batch_states)
            y_pred = predicted_q_vals[range(batch_size), batch_actions]

            return numpy_mse(y_pred, batch_targets)

        def pytorch_loss(data):
            batch_states, batch_actions, batch_targets = preprocess(policy, data)
            batch_size = batch_actions.shape[0]

            # Convert to tensors
            x_var = torch.tensor(batch_states, dtype=torch.float, device=policy.function_approximator.device)
            y_var = torch.tensor(batch_targets, dtype=torch.float, device=policy.function_approximator.device)

            predicted_q_vals = policy.function_approximator.eval_tensor(x_var)
            y_pred = predicted_q_vals[range(batch_size), batch_actions]

            return pytorch_mse(y_pred, y_var)

        if isinstance(policy, TabularPolicy):
            policy.compile(preprocess, self.learning_rate)
            return

        fa = policy.function_approximator
        assert not isinstance(fa, ACFunctionApproximator)

        if isinstance(fa, AbstractPyTorchFA):
            policy.compile(pytorch_loss, self.learning_rate)
        else:
            policy.compile(numpy_loss, self.learning_rate)

    @abc.abstractmethod
    def _get_action_value(self, policy: Policy, next_state: np.ndarray, next_action: np.ndarray):
        raise NotImplementedError()

    def update(self, policy: Policy, model: Model, trajectory: Trajectory):
        # Log trajectory
        if self.online:
            self.memory.log(trajectory)
        elif trajectory.done and not self.online:
            self.memory.log(trajectory)
        
        # Check update interval
        if self.memory.update and self.online and len(trajectory) > 1:
            # Sample memory
            batch = self.memory.sample()
            policy.update_with_dataset(batch)
        elif self.memory.update and not self.online and trajectory.done:
            # Sample memory
            batch = self.memory.sample()
            policy.update_with_dataset(batch)