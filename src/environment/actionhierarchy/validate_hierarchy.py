"""
Function for validating the format of a given action hierarchy
@author Dallan Goldblatt
"""
from common.predicates.predicates import validate_predicate


def validate_hierarchy(ah) -> bool:
    assert type(ah) == dict, 'The action hierarchy must be a dict'

    # The empty hierarchy is used for flat environments
    if ah == {}:
        return True

    assert 'state_variables' in ah, 'The action hierarchy must contain a list of state variables'
    state_variables = ah['state_variables']
    for state_variable in state_variables:
        assert type(state_variable) == str, 'Each state variable must be a string'

    assert 'primitive_action_map' in ah, 'The action hierarchy must contain a map of primitive actions to their ' \
                                         'action number in the environment'
    primitive_action_map = ah['primitive_action_map']
    assert type(primitive_action_map) == dict, 'The primitive action map mut be a dict'
    for key, value in primitive_action_map.items():
        assert type(value) == int, 'Each primitive action value must be an int'

    assert 'actions' in ah, 'The action hierarchy the dictionary of actions'
    actions = ah['actions']
    assert type(actions) == dict, 'actions must be a dict mapping each action name to its specification'
    assert 'Root' in actions, 'The action hierarchy must contain the action \'Root\''

    # Verify that each action has the correct list/dict/string structure
    for action, spec in actions.items():
        assert type(action) == str, 'Each action name in the hierarchy must be a string'
        assert type(spec) == dict, 'Each action specification in the hierarchy must be a dictionary'

        assert 'primitive' in spec, f'{action}\'s specification must contain the boolean \'primitive\''
        assert type(spec['primitive']) == bool, f'Action \'{action}\' must specify \'primitive\' as a bool'

        assert 'parents' in spec, f'{action}\'s specification must contain the list \'parents\''
        assert type(spec['parents']) == list, f'Action \'{action}\' must specify \'parents\' as a list'
        for parent in spec['parents']:
            assert parent in actions, f'Action \'{action}\' has parent \'{parent}\' that does not appear in ' \
                                          f'the hierarchy'

        if spec['primitive']:
            # Primitive actions only need to specify 'primitive' and 'parents'
            continue

        assert 'state_variables' in spec, f'{action}\'s action specification must contain the the list of relevant ' \
                                          f'state variables n \'state_variables\''
        assert type(spec['state_variables']) == list, f'Action \'{action}\' must specify \'state_variables\' as a list'
        for sv in spec['state_variables']:
            assert sv in state_variables, f'Action \'{action}\' has state variable \'{sv}\' that does not appear in ' \
                                          f'the global state variable list'

        assert 'params' in spec, f'{action}\'s specification must contain the dictionary \'params\''
        assert type(spec['params']) == dict, f'Action \'{action}\' must specify \'params\' as a dict'
        for param, bindings in spec['params'].items():
            assert type(bindings) == list, f'Parameter \'{param}\' for action \'{action}\' must specify the possible ' \
                                           f'bindings as a list'
            for binding in bindings:
                assert binding in state_variables, f'The binding \'{binding}\' to parameter \'{param}\' for action ' \
                                                   f'{action}\' does not appear in the global state variable list'

        assert 'children' in spec, f'{action}\'s specification must contain the dictionary \'children\''
        assert type(spec['children']) == dict, f'Action \'{action}\' must specify \'children\' as a dict'
        for child, params in spec['children'].items():
            assert type(params) == dict, f'{action}\'s child \'{child}\' must specify the parameter bindings as a dict'
            for param, binding in params.items():
                assert type(binding) == str, f'The binding to parameter \'{param}\' for child {child}\' of action ' \
                                             f'\'{action}\' must be a string '
                assert binding in state_variables, f'The binding \'{binding}\' to parameter \'{param}\' for child ' \
                                                   f'{child}\' of action \'{action}\' does not appear in the global ' \
                                                   f'state variable list'

        assert 'termination' in spec, f'{action}\'s specification must contain the list of termination predicates ' \
                                      f'\'termination\''
        assert type(spec['termination']) == list, f'Action \'{action}\' must specify \'termination\' as a list'
        for pred in spec['termination']:
            assert type(pred) == str, f'Action \'{action}\' must specify each termination predicate as a string'

        assert type(spec['pseudo_r']) == list, f'Action \'{action}\' must specify \'pseudo_r\' as a list of lists'
        for pr in spec['pseudo_r']:
            assert type(pr) == list, f'Action \'{action}\' must specify each pseudo reward as a list'
            if pr:
                assert len(pr) == 2, f'Each pseudo reward for Action \'{action}\' must contain a predicate and a value'
                assert type(pr[0]) == str, f'The first element of pseudo reward \'{pr}\' for Action \'{action}\' must' \
                                           f' be a predicate string'
                assert type(pr[1]) == int or type(pr[1]) == float, f'The second element of pseudo reward \'{pr}\' for' \
                                                                   f' Action \'{action}\' must be a numerical value'

    # Check action references are valid after validating structure
    for action, spec in actions.items():
        for parent in spec['parents']:
            # Check that this parent has the action as its child
            assert action in actions[parent]['children'], f'Action \'{action}\' has \'{parent}\' as its parent, but ' \
                                                          f'\'{parent}\' does not have  \'{action}\' as its child'

        if spec['primitive']:
            # Primitive actions do not have parameters
            continue

        for child, params in spec['children'].items():
            # Check that each child has this action as its parent
            assert action in actions[child]['parents'], f'Action \'{action}\' has \'{child}\' as its child, but ' \
                                                          f'\'{child}\' does not have  \'{action}\' as its parent'

            for param in params:
                assert param in actions[child]['params'], f'Action \'{action}\' passes an unexpected parameter ' \
                                                          f'\'{param}\' to its child \'{child}\''

            if not actions[child]['primitive']:
                for param in actions[child]['params']:
                    assert param in params, f'Action \'{action}\' does not pass required parameter \'{param}\' to its ' \
                                            f'child \'{child}\''

    # Check that predicates have correct syntax
    # There may still be semantical errors that are uncaught until runtime
    for action, spec in actions.items():
        if spec['primitive']:
            # Primitive actions do not have predicates
            continue

        # Create the list of variables that could be used in the predicates
        variables = [param for param in spec['params']]
        variables.extend(state_variables)

        for pred in spec['termination']:
            validate_predicate(pred, variables)

        for pr in spec['pseudo_r']:
            if pr:
                validate_predicate(pr[0], variables)

    return True
