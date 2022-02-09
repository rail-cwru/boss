import numpy as np

# To prevent divide by 0
EPSILON = 1e-10

def calc_discounted_rewards(discount_factor: float, rewards: np.ndarray, standardize: bool= False):
    episode_length = len(rewards)
    current_discount = 0
    discounted_rewards = np.zeros((episode_length))

    for i in reversed(range(0, episode_length)):
        current_discount = current_discount * discount_factor + rewards[i]
        discounted_rewards[i] = current_discount

    if standardize:
        # Normalize
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + EPSILON)
        discounted_rewards[discounted_rewards==0] = EPSILON

    return discounted_rewards