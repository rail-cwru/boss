import numpy as np
from collections import deque

class UCBValue(object):
    def __init__(self):
        self.expected_value = 0
        self.variance = 0
        self.use_count = 0

    def log(self, expected_value: float, variance: float):
        self.expected_value = expected_value
        self.variance = variance

    def calculate(self, alpha: float) -> float:
        if self.use_count == 0:
            return np.inf
        else:
            return self.expected_value + alpha * np.sqrt(self.variance)
    
    def update_use_count(self):
        self.use_count += 1

    def reset(self):
        self.expected_value = 0
        self.variance = 0
        self.use_count = 0

class UCB(object):
    def __init__(self, alpha: float, num_arms: int):
        self.alpha = alpha
        self.arms = [UCBValue() for x in range(num_arms)]

    def calculate(self, result_file: str, current_index: int) -> int:
        # Calculate values
        ucb_payoff = [0.0] * len(self.arms)
        for i_arm, arm in enumerate(self.arms):
            ucb_payoff[i_arm] = arm.calculate(self.alpha)

        # Write out UCB info
        ucb_result_file = '{}_{}.csv'.format(result_file, 'ucb')
        with open(ucb_result_file, mode='a') as f:
            row = [str(v) for v in ucb_payoff]
            row_str = ','.join(row)
            f.write(row_str + '\n')

        # Choose max value
        if np.inf in ucb_payoff:
            unchoosen_arms = [i for i, e in enumerate(ucb_payoff) if e == np.inf]
            chosen_index = np.random.choice(unchoosen_arms)
        else:
            # Find all maximum
            max_ucb = np.max(ucb_payoff)
            max_indices = [i for i,x in enumerate(ucb_payoff) if x == max_ucb]

            # If current index in tie, choose that
            if current_index in max_indices:
                chosen_index = current_index
            else:
                # Random tie break
                chosen_index = np.random.choice(max_indices)

        # Increment use count
        self.arms[chosen_index].update_use_count()

        return chosen_index

    def log(self, i_arm: int, expected_value: float, variance: float):
        self.arms[i_arm].log(expected_value, variance)

    def get_arm_value(self, i_arm: int) -> tuple:
        arm = self.arms[i_arm]
        return (arm.expected_value, arm.variance)

    def reset(self):
        for arm in self.arms:
            arm.reset()