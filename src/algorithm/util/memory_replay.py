import random
import numpy as np
from collections import deque

class MemoryReplay(object):
    def __init__(self, memory_size: int, batch_size: int, save_steps: bool, update_interval: int):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.save_steps = save_steps
        self.update_interval = update_interval
        self.steps = 0

    @property
    def update(self):
        return self.steps >= self.update_interval

    def sample(self):
        self.steps = 0
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        return batch

    def log(self, trajectory):
        self.steps += 1
        if self.save_steps:
            if len(trajectory) > 1:
                state = trajectory.observations[-2]
                action = trajectory.actions[-2]
                reward = trajectory.rewards[-2]
                next_state = trajectory.observations[-1]
                next_action = trajectory.actions[-1]
                step = (state, action, reward, False, next_state, next_action)

                if trajectory.done:
                    self.steps += 1
                    next_reward = trajectory.rewards[-1]
                    step = (next_state, next_action, next_reward, True, -1, -1)

                self.memory.append(step)
        else:
            traj_length = len(trajectory)
            states = np.zeros_like(trajectory.observations)
            actions = np.zeros_like(trajectory.actions)
            rewards = np.zeros((traj_length, 1))
            dones = np.zeros((traj_length, 1))
            next_states = np.zeros_like(trajectory.observations)
            next_actions = np.zeros_like(trajectory.actions)

            for i in range(traj_length):
                states[i,:] = trajectory.observations[i]
                actions[i,:] = trajectory.actions[i]
                rewards[i] = trajectory.rewards[i]
                
                if i == traj_length - 1:
                    dones[i] = True
                    next_states[i,:] = -1
                    next_actions[i,:] = -1
                else:
                    dones[i] = False
                    next_states[i,:] = trajectory.observations[i+1]
                    next_actions[i,:] = trajectory.actions[i+1]

            np_traj = (
                states, 
                actions, 
                rewards, 
                dones, 
                next_states, 
                next_actions
            )
            self.memory.append(np_traj)
