import numpy as np
from typing import List, Tuple, Any

from .ope_memory import ReplayTrajectory

# Supported Off policy estimators
ope_PF = 'PF'
ope_IS = 'IS'

# To prevent divide by 0
EPSILON = 1e-10

def CalculateIW(B: List[ReplayTrajectory]) -> Tuple[np.array, np.array]:
    # Get the number of trajectories
    n = len(B)

    # Importance weights and returns
    IWs = np.ones((n), dtype=float)
    R = np.zeros((n), dtype=float)
    for i in range(n):
        cur_prob_e = 1.0
        cur_prob_b = 1.0
        reward = 0.0
        for t in range(B[i].get_length()):
            cur_prob_e *= B[i].get_value('Evaluation Probabilities', t)
            cur_prob_b *= B[i].get_value('Behavior Probabilities', t)
            reward += B[i].get_value('Behavior Rewards', t)
 
        if cur_prob_b > EPSILON and cur_prob_e > EPSILON:
            IWs[i] = cur_prob_e / cur_prob_b
        else:
            IWs[i] = EPSILON
        R[i] = reward

    return IWs, R

"""
Importance sampling (IS) estimator. This can be weighted or unweighted.
"""
def IS(B: List[ReplayTrajectory], ope_params: List[Any]) -> float:
    # Parse parameters
    weighted_sampling = ope_params[0]

    # Get the number of trajectories
    n = float(len(B))

    # Importance weights and returns
    IWs, R = CalculateIW(B)

    # Mean
    if weighted_sampling:
        expected_reward = np.dot(IWs,R) / IWs.sum()
    else:
        expected_reward = np.dot(IWs,R) / n

    # Variance
    variance_values = np.zeros((n), dtype=float)
    for i in range(n):
        variance_values[i] = np.power((IWs[i] * R[i]) - expected_reward, 2.0)
    expected_var = variance_values.sum() / n

    return expected_reward, expected_var

"""
Particle filter (PF) estimator.
"""
def PF(B: List[ReplayTrajectory], ope_params: List[Any]) -> float:
    # Parse parameters
    resample_size = ope_params[0]

    # Importance weights and returns
    IWs, R = CalculateIW(B)
    IWs /= IWs.sum()

    # Construct and sample from distribution
    sampled_R = []
    resamples = np.random.multinomial(resample_size, IWs)
    for i in range(resamples.shape[0]):
        count = resamples[i]
        redundant_R = [R[i]] * count

        sampled_R.extend(redundant_R)

    # Mean and variance
    expected_reward = np.mean(sampled_R)
    expected_var = np.var(sampled_R)

    return expected_reward, expected_var

"""
KL Distance calculation
"""
def KL(new_mean: np.ndarray, new_sd: np.ndarray, old_mean: np.ndarray, old_sd: np.ndarray):
    new_sd_squared = np.power(new_sd, 2.0)
    mean_diff = old_mean - new_mean
    diff = np.power(old_sd, 2.0) + np.power(mean_diff, 2.0)

    approx_kl = np.log(new_sd / old_sd) + diff /  (2.0 * new_sd_squared + 10e-8) - 0.5
    approx_kl = np.sum(approx_kl, axis=1)
    approx_kl = np.mean(approx_kl)
    return approx_kl