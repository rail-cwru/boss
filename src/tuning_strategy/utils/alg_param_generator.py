import copy
import numpy as np
from typing import List

from config import Config
from algorithm import Algorithm, QLearning, SARSA, Reinforce, A2C
from policy import TabularPolicy, FuncApproxPolicy
from .off_policy_learner import OffPolicyLearner

# Algorithm choices
Q_LEARN_NAME = 'QLearning'
REINFORCE_NAME = 'Reinforce'
SARSA_NAME = 'SARSA'
A2C_NAME = 'A2C'

discount_factor_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]  # Discount Factor
learning_rate_range = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1]   # Learning Rate
entropy_coeff_range = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
value_coeff_range = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# Policy sampler choices
THOMPSON = 'DiscreteBoltzmann'
EPSILON_GREEDY = 'DiscreteEGreedy'

temperature_range = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2]
epsilon_range = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
min_epsilon_range = [0.01, 0.05, 0.1, 0.2, 0.3]
decay_range = [0.9, 0.95, 0.99, 0.995, 0.999]

def perturb_param_set(alg_config: Config, policy_config: Config, perturb_factor: float, tune_policy_sampler_params: bool):
    lower_bound = 1-perturb_factor
    upper_bound = 1+perturb_factor

    # Peterb alg params
    if alg_config.name == Q_LEARN_NAME or alg_config.name == SARSA_NAME:
        alg_config.discount_factor *= np.random.uniform(lower_bound, upper_bound)
        alg_config.learning_rate *= np.random.uniform(lower_bound, upper_bound)

        # Check params
        if alg_config.discount_factor <= 0.0:
            alg_config.discount_factor = min(discount_factor_range)
        elif alg_config.discount_factor > 1.0:
            alg_config.discount_factor = max(discount_factor_range)
    elif alg_config.name == REINFORCE_NAME:
        alg_config.discount_factor *= np.random.uniform(lower_bound, upper_bound)
        alg_config.learning_rate *= np.random.uniform(lower_bound, upper_bound)

        # Check params
        if alg_config.discount_factor <= 0.0:
            alg_config.discount_factor = min(discount_factor_range)
        elif alg_config.discount_factor > 1.0:
            alg_config.discount_factor = max(discount_factor_range)
    elif alg_config.name == A2C_NAME:
        alg_config.discount_factor *= np.random.uniform(lower_bound, upper_bound)
        alg_config.learning_rate *= np.random.uniform(lower_bound, upper_bound)
        alg_config.entropy_coeff *= np.random.uniform(lower_bound, upper_bound)
        alg_config.value_coeff *= np.random.uniform(lower_bound, upper_bound)

        # Check params
        if alg_config.discount_factor <= 0.0:
            alg_config.discount_factor = min(discount_factor_range)
        elif alg_config.discount_factor > 1.0:
            alg_config.discount_factor = max(discount_factor_range)

        if alg_config.entropy_coeff < 0.0:
            alg_config.entropy_coeff = min(entropy_coeff_range)
        elif alg_config.entropy_coeff > 1.0:
            alg_config.entropy_coeff = max(entropy_coeff_range)

        if alg_config.value_coeff < 0.0:
            alg_config.value_coeff = min(value_coeff_range)
        elif alg_config.value_coeff > 1.0:
            alg_config.value_coeff = max(value_coeff_range)
    else:
        raise Exception('Unknown RL aglorithm: {}'.format(alg_config.name))


    # Peterb sampler params if required
    if tune_policy_sampler_params:
        if policy_config.policy_sampler.name == THOMPSON:
            policy_config.policy_sampler.temperature *= np.random.uniform(lower_bound, upper_bound)
        elif policy_config.policy_sampler.name == EPSILON_GREEDY:
            if policy_config.policy_sampler.decay == 1.0:
                policy_config.policy_sampler.epsilon *= np.random.uniform(lower_bound, upper_bound)

                # Check params
                if policy_config.policy_sampler.epsilon <= 0.0:
                    policy_config.policy_sampler.epsilon = min(epsilon_range)
                elif policy_config.policy_sampler.epsilon >= 1.0:
                    policy_config.policy_sampler.epsilon = max(epsilon_range)
            else:
                policy_config.policy_sampler.min_epsilon *= np.random.uniform(lower_bound, upper_bound)
                policy_config.policy_sampler.decay *= np.random.uniform(lower_bound, upper_bound)

                # Check params
                if policy_config.policy_sampler.decay <= 0.0:
                    policy_config.policy_sampler.decay = min(decay_range)
                elif policy_config.policy_sampler.decay > 1.0:
                    policy_config.policy_sampler.decay = max(decay_range)

                if policy_config.policy_sampler.min_epsilon <= 0.0:
                    policy_config.policy_sampler.min_epsilon = min(min_epsilon_range)
                elif policy_config.policy_sampler.min_epsilon > 1.0:
                    policy_config.policy_sampler.min_epsilon = max(min_epsilon_range)

    return alg_config, policy_config

def generate_param_set(ensemble_size: int, baseline_alg_config: Config, baseline_policy_config: Config):
    possible_alg_param_sets, baseline_alg_index = __get_alg_param_set_indicies(baseline_alg_config)
    possible_policy_param_sets, baseline_policy_index = __get_policy_param_set_indicies(baseline_policy_config)
    num_alg_sets = len(possible_alg_param_sets)
    num_policy_sets = len(possible_policy_param_sets)

    # Generate indicies of configurations and shuffle
    total_configs =  num_alg_sets * num_policy_sets
    overall_baseline_index = (baseline_alg_index * num_policy_sets) + baseline_policy_index
    possible_param_indices = list(range(total_configs))
    np.random.shuffle(possible_param_indices)

    # Select subset
    param_sample_indices = np.random.choice(possible_param_indices, ensemble_size, replace=False)

    # Ensure basline is included in sample
    if overall_baseline_index not in param_sample_indices:
        param_sample_indices[0] = overall_baseline_index
        baseline_final_index = 0
    else:
        baseline_final_index = np.where(param_sample_indices==overall_baseline_index)

    # Retrieve configs
    alg_param_samples = []
    policy_param_samples = []
    for param_sample in param_sample_indices:
        alg_param_sample_index = np.floor(param_sample / num_policy_sets).astype('int')
        policy_param_sample_index = param_sample % num_policy_sets

        alg_param_samples.append(possible_alg_param_sets[alg_param_sample_index])
        policy_param_samples.append(possible_policy_param_sets[policy_param_sample_index])

    # Construct configs
    alg_configs = __construct_alg_configs(baseline_alg_config, alg_param_samples)
    policy_configs = __construct_policy_configs(baseline_policy_config, policy_param_samples)

    return zip(alg_configs, policy_configs), baseline_final_index

def draw_param_set(alg_config: Config, policy_config: Config, tune_policy_sampler_params: bool):
    if alg_config.name == Q_LEARN_NAME or alg_config.name == SARSA_NAME:
        alg_config.discount_factor = np.random.choice(discount_factor_range)
        alg_config.learning_rate = np.random.choice(learning_rate_range)
    elif alg_config.name == REINFORCE_NAME:
        alg_config.discount_factor = np.random.choice(discount_factor_range)
        alg_config.learning_rate = np.random.choice(learning_rate_range)
    elif alg_config.name == A2C_NAME:
        alg_config.discount_factor = np.random.choice(discount_factor_range)
        alg_config.learning_rate = np.random.choice(learning_rate_range)
        alg_config.entropy_coeff = np.random.choice(entropy_coeff_range)
        alg_config.value_coeff = np.random.choice(value_coeff_range)
    else:
        raise Exception('Unknown RL aglorithm: {}'.format(alg_config.name))

    # Peterb sampler params if required
    if tune_policy_sampler_params:
        if policy_config.policy_sampler.name == THOMPSON:
            policy_config.policy_sampler.temperature = np.random.choice(temperature_range)
        elif policy_config.policy_sampler.name == EPSILON_GREEDY:
            if policy_config.policy_sampler.decay == 1.0:
                policy_config.policy_sampler.epsilon = np.random.choice(epsilon_range)
            else:
                policy_config.policy_sampler.epsilon = 1.0
                policy_config.policy_sampler.min_epsilon = np.random.choice(min_epsilon_range)
                policy_config.policy_sampler.decay = np.random.choice(decay_range)

    return alg_config, policy_config

def __get_alg_param_set_indicies(baseline_alg_config: Config):
    possible_param_sets = []
    baseline_index = -1
    if baseline_alg_config.name == Q_LEARN_NAME or baseline_alg_config.name == SARSA_NAME:
        for df in discount_factor_range:
            for lr in learning_rate_range:
                possible_param_sets.append((df, lr))

                if df == baseline_alg_config.discount_factor and lr == baseline_alg_config.learning_rate:
                    baseline_index = len(possible_param_sets) - 1

    elif baseline_alg_config.name == REINFORCE_NAME:
        for df in discount_factor_range:
            for lr in learning_rate_range:
                possible_param_sets.append((df, lr))

                if df == baseline_alg_config.discount_factor and lr == baseline_alg_config.learning_rate:
                    baseline_index = len(possible_param_sets) - 1
    elif baseline_alg_config.name == A2C_NAME:
        for df in discount_factor_range:
            for lr in learning_rate_range:
                for ec in entropy_coeff_range:
                    for vc in value_coeff_range:
                        possible_param_sets.append((df, lr, ec, vc))

                        if df == baseline_alg_config.discount_factor and lr == baseline_alg_config.learning_rate and ec == baseline_alg_config.entropy_coeff and vc == baseline_alg_config.value_coeff:
                            baseline_index = len(possible_param_sets) - 1
    else:
        raise Exception('Unknown RL aglorithm: {}'.format(baseline_alg_config.name))

    return possible_param_sets, baseline_index

def __get_policy_param_set_indicies(baseline_policy_config: Config):
    possible_param_sets = []
    baseline_index = -1
    
    if baseline_policy_config.policy_sampler.name == THOMPSON:
        for temp in temperature_range:
            possible_param_sets.append(temp)

            if temp == baseline_policy_config.policy_sampler.temperature:
                baseline_index = len(possible_param_sets) - 1

    elif baseline_policy_config.policy_sampler.name == EPSILON_GREEDY:
        if baseline_policy_config.policy_sampler.decay == 1.0:
            for eps in epsilon_range:
                possible_param_sets.append((eps, baseline_policy_config.policy_sampler.min_epsilon, baseline_policy_config.policy_sampler.decay))

                if eps == baseline_policy_config.policy_sampler.epsilon:
                    baseline_index = len(possible_param_sets) - 1
        else:
            for eps_min in min_epsilon_range:
                for eps_decay in decay_range:
                    possible_param_sets.append((baseline_policy_config.policy_sampler.epsilon, eps_min, eps_decay))

                    if eps_min == baseline_policy_config.policy_sampler.min_epsilon and eps_decay == baseline_policy_config.policy_sampler.decay:
                        baseline_index = len(possible_param_sets) - 1
    else:
        # Default param baseline
        possible_param_sets.append(0)
        baseline_index = 0

    return possible_param_sets, baseline_index

def __construct_alg_configs(baseline_alg_config: Config, alg_param_samples):
    configs = []
    if baseline_alg_config.name == Q_LEARN_NAME:
        # Create data dictionary
        for discount_factor, learning_rate in alg_param_samples:
            cfg_dict = {}
            cfg_dict['name'] = baseline_alg_config.name
            cfg_dict['memory_size'] = baseline_alg_config.memory_size
            cfg_dict['batch_size'] = baseline_alg_config.batch_size
            cfg_dict['update_interval'] = baseline_alg_config.update_interval
            cfg_dict['discount_factor'] = discount_factor
            cfg_dict['learning_rate'] = learning_rate
            cfg_dict['is_online'] = baseline_alg_config.is_online

            cfg = Config(QLearning, cfg_dict, top_level=False)
            configs.append(cfg)
    elif baseline_alg_config.name == SARSA_NAME:
        # Create data dictionary
        for discount_factor, learning_rate in alg_param_samples:
            cfg_dict = {}
            cfg_dict['name'] = baseline_alg_config.name
            cfg_dict['memory_size'] = baseline_alg_config.memory_size
            cfg_dict['batch_size'] = baseline_alg_config.batch_size
            cfg_dict['update_interval'] = baseline_alg_config.update_interval
            cfg_dict['discount_factor'] = discount_factor
            cfg_dict['learning_rate'] = learning_rate
            cfg_dict['is_online'] = baseline_alg_config.is_online

            cfg = Config(SARSA, cfg_dict, top_level=False)
            configs.append(cfg)
    elif baseline_alg_config.name == REINFORCE_NAME:
        for discount_factor, learning_rate in alg_param_samples:
            cfg_dict = {}
            cfg_dict['name'] = baseline_alg_config.name
            cfg_dict['memory_size'] = baseline_alg_config.memory_size
            cfg_dict['batch_size'] = baseline_alg_config.batch_size
            cfg_dict['update_interval'] = baseline_alg_config.update_interval
            cfg_dict['discount_factor'] = discount_factor
            cfg_dict['learning_rate'] = learning_rate

            cfg = Config(Reinforce, cfg_dict, top_level=False)
            configs.append(cfg)
    elif baseline_alg_config.name == A2C_NAME:
        for discount_factor, learning_rate, entropy_coeff, value_coeff in alg_param_samples:
            cfg_dict = {}
            cfg_dict['name'] = baseline_alg_config.name
            cfg_dict['memory_size'] = baseline_alg_config.memory_size
            cfg_dict['batch_size'] = baseline_alg_config.batch_size
            cfg_dict['update_interval'] = baseline_alg_config.update_interval
            cfg_dict['discount_factor'] = discount_factor
            cfg_dict['learning_rate'] = learning_rate
            cfg_dict['entropy_coeff'] = entropy_coeff
            cfg_dict['value_coeff'] = value_coeff

            cfg = Config(A2C, cfg_dict, top_level=False)
            configs.append(cfg)
    else:
        raise Exception('Unknown RL aglorithm: {}'.format(baseline_alg_config.name))

    return configs

def __construct_policy_configs(baseline_policy_config: Config, policy_param_samples):
    configs = []
    if baseline_policy_config.policy_sampler.name == THOMPSON:
        for temp in policy_param_samples:
            cfg = copy.deepcopy(baseline_policy_config)
            cfg.policy_sampler.temperature = temp

            configs.append(cfg)
    elif baseline_policy_config.policy_sampler.name == EPSILON_GREEDY:
        for eps, eps_min, eps_decay in policy_param_samples:
            cfg = copy.deepcopy(baseline_policy_config)
            cfg.policy_sampler.epsilon = eps
            cfg.policy_sampler.min_epsilon = eps_min
            cfg.policy_sampler.decay = eps_decay

            configs.append(cfg)
    else:
        for _ in policy_param_samples:
            # Copy baseline and add
            cfg = copy.deepcopy(baseline_policy_config)
            configs.append(cfg)

    return configs