"""
Base config module.

Guidelines:

Config objects should be JSON and stored in experiment_configs.
    No config should contain nested configs as they are in the JSON.
        Nested dictionaries are ok only if they are pure "primitive" data: Int maps, etc.
    Nesting of standard module configs should be described by listing file paths.

Two special fields may be in each config object:
    'base' which contains a file path of a base config.
    Any modifications in the config file will override the imported base config.

    For example, one might create a config for a modified PredatorPrey environment,
    which uses as base config the config file for the original PredatorPrey environment.


Environment config should have a dict of agent class to count of agents in that class.

"""
from .config import Config, ConfigItemDesc, ConfigDesc, AbstractModuleFrame
