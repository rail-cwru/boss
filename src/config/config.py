"""
Config class.

The config object itself.
Used to define replicable experiment parameters.


JSON Runner Pipeline:
    SPECIFICATION:              Recursive "path" / "base" / "name" specification
    CLASSLOADING + LOADING:     Classes loaded via Config through


Script Runner Pipeline:
    SPECIFICATION:              Additionally allows literal module_class specification
    CLASSLOADING + TRANSLATION: Data is translated into dicts with module class
    CONFIG CONSTRUCTION:        Dicts with module class loaded into config object


GUI Runner Pipeline:
    SPECIFICATION:              Script runner contents but through GUI


Notes:
    "None" in config is "null"
    Tuples don't exist in JSON. Use Lists.
"""
import keyword
import os
import json
import importlib

from collections import OrderedDict
from typing import Dict, Any, Callable, List, Type

from common.properties import Properties
from config.moduleframe import AbstractModuleFrame


class Config(object):

    def __init__(self,
                 module_class: Type[AbstractModuleFrame],
                 info_dict: Dict[str, Any],
                 relative_path: str= './',
                 top_level=True):
        """
        A configuration object which contains realized (non-dict) config items or objects.

        Each configuration object necessarily comes from a ConfigDesc, and thus contains a field [module_class]
            which points to the referenced class.

        ConfigItemDesc will correspond only to the attributes of a Config.

        Top-level Configs should be those of Controllers.

        :param module_class: The class of the module for which the Config is generated.
        :param info_dict: The dictionary containing the non-loaded info tied to this module class.
        :param relative_path: Relative path used for loading configs from files.
        :param top_level: Whether the config is NOT created by another config
        """
        assert issubclass(module_class, AbstractModuleFrame), '[{}] is not a valid usable module.'.format(module_class)
        self.module_class = module_class
        self.name = module_class.__name__

        props_to_check: List[Properties] = [module_class.properties()]

        # Explicit dictionary that points specifically to subconfigs
        self.subconfigs = {}

        # check and load attributes
        known_fields = {'name', 'module_class', '__relative_path'}
        for item in self.module_class.get_class_config():
            try:
                value = item.process(info_dict[item.name], relative_path)
            except KeyError:
                if item.optional:
                    value = item.default
                else:
                    raise ValueError('Did not find required config [{}] for module [{}]'.format(item.name, self.name))

            try:
                item.check(value)
            except AssertionError as e:
                print('The item [{}] was invalid for module [{}].'.format(item.name, self.name))
                raise e

            # If item is a ConfigDesc or subclass thereof, data should be made into Config
            if isinstance(item, ConfigDesc):
                # TODO clean up
                if type(item) == ConfigDesc:
                    # TODO Ensure that finding the right class is possible (no duplicates)
                    value = Config(value['module_class'], value, value['__relative_path'], top_level=False)
                    self.subconfigs[value.name] = value
                    props_to_check.append(value.properties)
                elif type(item) == ConfigListDesc:
                    new_value = OrderedDict()
                    for k, v in value.items():
                        new_value[k] = Config(v['module_class'], v, v['__relative_path'], top_level=False)
                        props_to_check.append(new_value[k].properties)
                    value = new_value
            setattr(self, item.name, value)
            known_fields.add(item.name)
        for k in info_dict:
            if k not in known_fields:
                print('Unknown field [{}] in config for [{}].'.format(k, self.name))

        composite_properties = Properties()
        for properties in props_to_check:
            composite_properties.merge(properties)
        if top_level:
            composite_properties.check_required_are_satisfied()
        self.properties = composite_properties

        # TODO add a function that returns the class-specific module config for a module instance passed in as argument

    # We will need to load the hirearchy config
    # And modify the loading function parse the hirearchial json files
    # THis class is called from run_with_config and generates data
    def load_hirearchy(self):
        pass



    def find_config_for_class(self, cls: Type):
        """
        Return the subconfig belonging to a class.
        Traverses nested configurations in pre-order.
        If class is not found, raises ValueError.

        TODO (urgent) behavior when multiple matching classes is ill-defined. Address when possible.
        TODO (urgent) circular references not handled (trivial to implement recursion check)

        :param cls: Class requested
        :return: Configuration
        :raises ValueError if class not found
        """
        target = cls.__name__
        # print('Target {}'.format(target))
        visited = set()
        if self.name == target:
            return self
        stack = [(module_name, subconfig) for module_name, subconfig in self.subconfigs.items()]
        # print(target, "wasn't right, looking in stack...", [x[0] for x in stack])
        while stack:
            module_name, subconfig = stack.pop()
            visited.add(subconfig)
            # print('Look in', module_name, len(stack))
            if target == module_name:
                # print('Found config,', subconfig)
                return subconfig
            else:
                stack.extend([(module_name, subconfig) for module_name, subconfig in subconfig.subconfigs.items()])
                # print(module_name, "wrong, check", stack)
        raise ValueError('The config did not contain the requested class.')

    def find_config_for_instance(self, obj: Any):
        """
        Return the subconfig belonging to an instance by calling :func:`~config.Config.find_config_for_class`.
        """
        return self.find_config_for_class(type(obj))


class ConfigItemDesc(object):

    def __init__(self, name: str, check: Callable, info: str, optional: bool=False, nestable=False, default=None):
        """
        Initialize a configuration item's description.
        :param name: Name of the configuration item.
        :param check: Function which checks the validity of the item once instantiated or loaded.
        :param info: A helpful description associated with the item.
        :param optional: Whether this item is optional.
        :param nestable: Whether this item can be loaded via another json
            (If True, you can specify a json file to load it by instead of the field necessarily from the file directly)
        :param default: Default value for item to fall back to if optional.
        """
        self.name = name
        self.info = info
        self.nestable = nestable
        self.optional = optional
        self.default = default

        self.__check = check
        self.__repr_str = self.name + ': ' + self.info
        # Validate name is a valid python identifier
        assert name.isidentifier() and not keyword.iskeyword(name), '[{}] is not a valid config item name.\n' \
                                                                    'A config name must be a valid attribute name.'
        assert callable(check), 'Check function must be callable.'
        if optional:
            assert default is not None, 'Optional ConfigItemDesc must have a default value set.'
        if default is not None:
            assert optional, 'ConfigItemDesc cannot have default value with non-optional config item.' \
                             '\nEither make the item optional or remove the default value.'

    def check(self, value) -> bool:
        """
        Checks the value of the inputted data to make sure it is a valid fit.

        May also use assertion errors in addition to the required return of a boolean indicating fit or failure.
        :param value: Value to set the configuration to.
        :return: Boolean of true or false.
        """
        return self.__check(value)

    def process(self, entry, relative_path=None, **kwargs):
        """
        Perform sanitization or additional processing on input data.
        :param entry: Name of configured item
        """
        if self.nestable:
            module_json_file = os.path.join(relative_path, entry)
            entry = json.load(open(module_json_file, 'r'))
        return entry

    def __repr__(self):
        return self.__repr_str


class ConfigDesc(ConfigItemDesc):

    def __init__(self, name: str, info: str, module_package: str = None, default_configs: List[str]=None):
        """
        Create a configuration description for a nested configuration (which corresponds to a sub-module).
        :param name : Name of config
        :param module_package: The package the sub-module should be imported from.
                                The default name for the required config field will be the lowest-level package name,
                                e.g. if the package is mypackage.mysubpackage, then the expected config field key will
                                be "mysubpackage".
        :param info: Information about the config and usually the contextual purpose of the module.
        :param default_configs: "Typical" default config files.
        """
        super().__init__(name, lambda: True, info)
        if module_package is None:
            self.module_package = name
        else:
            self.module_package = module_package
        self.default_configs = [] if default_configs is None else default_configs

    def check(self, data) -> bool:
        """
        Check that the data is a valid unloaded config data dict / path to file w/o checking existence of module.
        """
        ConfigDesc._module_check(data)
        return True

    def process(self, entry, relative_path=None, **kwargs):
        """
        Load data from json / sanitize data dictionary to correct form
        """
        module_config_dict = self._sanitize_and_load(entry, self.module_package, relative_path)
        return module_config_dict

    @staticmethod
    def _module_check(data):
        assert isinstance(data, dict), 'Sanitized module config should be a dictionary'
        assert 'module_class' in data, 'Sanitized module config should include a field "module_config"' \
                                       ' containing the loaded class of the target module.'

    @staticmethod
    def _sanitize_and_load(entry, package, relative_path):
        """
        Attempt to load the module specified by the config.
        :param entry: Name of item
        :param package: Package name
        :param relative_path: Relative config path
        :return:
        """
        try:
            # Option 1: Just specified the file path to the subconfig
            if not isinstance(entry, dict):
                assert isinstance(entry, str), "Invalid config for [{}]." \
                                               "Expected a filepath or appropriate dictionary for quick overrides." \
                                               "If you need an example, check example_experiment_1.json" \
                                               "or example_experiment_2.json, and read the documentation.".format(entry)
                module_json_file = os.path.join(relative_path, entry)
                err_msg = "Couldn't find the module config for [{}] in [{}].\n" \
                          "Make sure that the path to the config JSON is specified relative " \
                          "to the location of the experiment config.".format(package, module_json_file)
                assert os.path.exists(module_json_file), err_msg
                module_config_dict = json.load(open(module_json_file, 'r'))
                module_config_dict['__relative_path'] = os.path.dirname(module_json_file)
            else:
                # Option 2: Dictionary for quick overriding
                if 'base' in entry:
                    module_json_file = os.path.join(relative_path, entry['base'])
                    module_config_dict = json.load(open(module_json_file, 'r'))
                    module_config_dict['__relative_path'] = os.path.dirname(module_json_file)
                    del entry['base']
                # Option 3: Dictionary for quick definition
                # Option 4: Manual class-loading from config specified in script
                elif 'name' in entry or 'module_class' in entry:
                    # Will get checked in "check" function
                    module_config_dict = entry
                    if 'name' not in entry:
                        module_config_dict['name'] = module_config_dict['module_class'].__name__
                else:
                    raise ValueError('The provided config entry [{}] for module [{}] was invalid.\n'
                                     'Please check the config to check for the error.'.format(entry, package))
                # Overrides / quick definition
                for k, v in entry.items():
                    module_config_dict[k] = v
        except Exception as e:
            print('Encountered an exception while loading config for [{}]:'.format(package))
            raise e
        assert 'name' in module_config_dict, 'Configuration dictionaries must always contain the "name" field ' \
                                             'which points to the module being loaded. Check that the field' \
                                             ' is present and that configs are satisfied.'
        if 'module_class' not in module_config_dict:
            module_name = module_config_dict['name']
            prefixed_module_name = '.' + module_name
            module_class = getattr(importlib.import_module(prefixed_module_name, package=package), module_name)
            module_config_dict['module_class'] = module_class
        if '__relative_path' not in module_config_dict:
            module_config_dict['__relative_path'] = relative_path
        return module_config_dict


class ConfigListDesc(ConfigDesc):
    """
    Describes config for lists of modules (e.g. a controller having a list of callbacks).

    Names are automatically taken from the "name" fields of the sub-configs.

    Data should be an collections.OrderedDict mapping name to config Config object.
    """

    def check(self, ordered_dict):
        for datum in ordered_dict.values():
            ConfigDesc._module_check(datum)

    def process(self, entry, relative_path=None, **kwargs):
        ret = OrderedDict()
        for datum in entry:
            datum = ConfigDesc._sanitize_and_load(datum, self.module_package, relative_path)
            ret[datum['name']] = datum
        return ret
