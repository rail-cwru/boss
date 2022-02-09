import os
import random
import numpy as np
from torch import manual_seed, get_rng_state, set_rng_state


def ensure_file_writable(file: str) -> str:
    """
    Ensures that the file-path is a writable location (i.e. that the directory exists)
    """
    if file and not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    return file


def set_initial_seed(use_pytorch: bool, seed: int):
    # For repeatability
    random.seed(seed)
    np.random.seed(seed)

    if use_pytorch:
        manual_seed(seed)

def get_seed_state(use_pytorch: bool, env: 'Environment'):
    rnd_seed_state = random.getstate()
    np_seed_state = np.random.get_state()
    env_seed_state = env.get_seed_state()
    seed_state = [
        rnd_seed_state,
        np_seed_state, 
        env_seed_state
    ]

    if use_pytorch:
        pytorch_seed_state = get_rng_state()
        seed_state.append(pytorch_seed_state)

    return seed_state

def set_seed_state(use_pytorch: bool, env: 'Environment', seed_state):
    random.setstate(seed_state[0])
    np.random.set_state(seed_state[1])
    env.set_seed_state(seed_state[2])

    if use_pytorch:
        set_rng_state(seed_state[3])