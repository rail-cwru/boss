# BOSS 

This is an RL Framework with both online and offline capabilities. 
That said, it is made available to accompany a paper on offline hierarchical RL. 

This is a modular and extensible one-stop framework for Reinforcement Learning with Agent interaction schemes.

For requirements see `src/requirements.txt`

## Running
For all analyses, the config must have the following fields:
-Environment
-Agent System
-Policy
-Algorithm

### Offline analysis
For offline analysis, the configuration file also needs a sampler field. 
The easiest way to run an offline analysis is to use the following fields:
-samples_target tells you how many samples to collect
-episode_max_length: how long an episode can take 
-eval_num_samples: tells the sampler at which sample thresholds to analyze the model
-samples_name: prefix of saved files

Each sampler has its own hyper-parameters amd conigurations. For example, the FlattenedSamplers are expecting the user to specify both the flattener (BUF or TDF) and the number of iterations to perform.
Please see the examples in the config directory for more concrete examples.

### Configuration Files:

All runs accept the following arguments in the config file:
- episodes
- episode_max_length
- environment
- Agent System
- Algorithm
- Callbacks
- samples_name(for saving)


The offline controller is expecting several outputs that ub addition to those for online learning. Here are some of the important ones:
- kl_divergence: will plot kl divergence for each sample threshold in this list. The target distribution is to be passed into the sampler as "save_name" (this also is how you save a distribution:
- novel_states_count: Tracks the number of novel states encountered as a function of samples collected
- save_samples: Will save the list of samples for future use (list may be large!).
- keep_fraction: Will remove a fraction of the collected samples. Has been shown to be effective at improving runtime with little to no effect on rate of convergence.
- samples_target: Instead of specifying the number of episodes to run, the offline controller allows the user to specify the number of samples to collect 
- eval_num_samples: The user can specify the number of samples at which to analyze the model (This tends to be the preferred method).
- eval_samples: The user can specify the number of episodes after which to analyze the model. 


### BOSS Analysis
In addition to the offline fields, a BOSS analysis must also have a *steps_per_sampler* field, which determines the number of samples collected in the initial collection phase. 
Note that the *eval_num_samples* list that will dictate when the sampler exists this initial collection phase, so it is recommended that the first entry in this list equals *num_samplers x steps_per_sampler*.
Once the BOSS sampler leaves the intial learning phase, the *steps_per_sampler* field will no longer be used. 
The BOSS algorithm will stop at every instance in *eval_num_samples* and select a new sampler. Note that this list must be monotonically increasing. 


## Output

The framework will create a directory for saving results. The name of the directory is dictated by the *samples_name* field in the config. 
In this directory, the rewards (eval), cumulative_max (cummax), runtime (rt), and number of samples collected at each analysis point (len) are saved in the directory.
These values are saved both as .txt and .pkl and are updated and overwritten every iteration. 


### BOSS output
The BOSS analyses will output a few additional pieces of information. 
-First, it will output the list of samplers that were selected at each point in the analysis. 
-It will also plot the BOSS reward vs all biased samplers as a function of length for each run. 
-It will also plot the UCB of each sampler to illustrate decision making for each run. 
-The reward of each sampler will be saved in a different txt and pkl file. These files hold a 2-d list, which is a list of reward lists, where each list represents the rewards acheived by the policy trained on the samples collected with that sampler. The runtime, len, eval and cummax are all still stored, and these represent the rewards of the policy trained on all of the samples collected (i.e the reward of the overall BOSS sampler)

