from typing import List, Dict, Any

class ReplayTrajectory(object):
    def __init__(self, record_keys: List[str]):
        self._len = 0
        self._records = dict([(key, []) for key in record_keys])

    def add_step(self, steps: Dict[str, Any]):
        for key, value in steps.items():
            self._records[key].append(value)
        self._len += 1

    def get_value(self, key: str, index):
        return self._records[key][index]

    def get_length(self):
        return self._len

class OpeMemory(object):
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.memory_index = -1
        self.behavior_memory = []
        self.behavioral_keys = ['States', 'Actions', 'Rewards']
        self.ope_keys = ['Behavior Probabilities', 'Behavior Rewards', 'Evaluation Probabilities']

    def get_ope_trajectories(self, behav_algorithm, eval_algorithm, pg_id):
        D = []
        for behav_record in self.behavior_memory:
            ope_record = ReplayTrajectory(self.ope_keys)

            for j_step in range(behav_record.get_length()):
                # Get behavior values
                behav_state = behav_record.get_value('States', j_step)
                behav_action = behav_record.get_value('Actions', j_step)
                behav_reward = behav_record.get_value('Rewards', j_step)

                # Get behavior probability
                behav_action_probs = behav_algorithm.get_action_prob(behav_state)
                behav_probability = behav_action_probs[pg_id][behav_action]

                # Get evaluation probability
                eval_action_probs = eval_algorithm.get_action_prob(behav_state)
                eval_probability = eval_action_probs[pg_id][behav_action]

                # Record
                ope_vals = dict(zip(self.ope_keys, [behav_probability, behav_reward, eval_probability]))
                ope_record.add_step(ope_vals)

            D.append(ope_record)

        return D

    def log(self, trajectory):
        for i_step in range(len(trajectory)):
            state = trajectory.observations[i_step]
            action = trajectory.actions[i_step][0]
            reward = trajectory.rewards[i_step]

            # Initialize new record if first step
            if i_step == 0:
                if self.memory_index == self.memory_size - 1:
                    self.memory_index = 0
                else:
                    self.memory_index += 1

                # Update behavioral memory for opes
                if len(self.behavior_memory) < self.memory_size:
                    self.behavior_memory.append(ReplayTrajectory(self.behavioral_keys))
                else:
                    self.behavior_memory[self.memory_index] = ReplayTrajectory(self.behavioral_keys)

            # Record behavioral data
            replay_vals = dict(zip(self.behavioral_keys, [state, action, reward]))
            self.behavior_memory[self.memory_index].add_step(replay_vals)

    def reset(self):
        self.memory_index = -1
        self.behavior_memory = []