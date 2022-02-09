"""
Function for evaluating predicates
@author Dallan Goldblatt
"""
import numpy as np
import re
from typing import Dict, Any, List

import ply.lex as lex
import ply.yacc as yacc

from common.predicates import predicate_yacc, predicate_lex
from common.predicates.functions import dispatcher
from domain import ObservationDomain


def validate_predicate(predicate: str,
                       variables: List[str]) -> bool:
    reserved = ['AND', 'OR', 'NOT', 'TRUE', 'FALSE']

    # Make sure that the variables in the predicate are in the variables parameter
    matches = re.findall(r'[a-zA-Z_][a-zA-Z_0-9]*', predicate)

    for match in matches:
        valid = match in reserved or match in variables or match in dispatcher
        assert valid, f'The predicate \'{predicate}\' references an unknown variable \'{match}\''

    return True


def evaluate_predicate(predicates: List[str],
                       obs_domain: ObservationDomain,
                       state: np.ndarray,
                       additional_vars: Dict[str, Any]
                       ) -> bool:
    """
    Entry point for the predicate evaluator.
    Evaluates whether a logical predicate is satisfied in the current state.
    @author Dallan Goldblatt
    :param predicates: A list of strings containing correctly formatted predicates
    :param obs_domain: The observation domain state refers to
    :param state: An observation domain received from the environment
    :param additional_vars: Extra variables that may be referenced by name in the predicate
    :return: boolean indicating if the predicate is satisfied in the passed state
    """
    # Extract the names and values of each state variable
    indexed_vars = {}
    for item in obs_domain.items:
        value = obs_domain.get_item_view_by_name(state, item.name)
        if item.shape == [1]:
            # This var is only one item, extract its value
            value = value[0]
        else:
            # Convert the np.array to a list
            value = value.tolist()

        if not item.prefix:
            # This state var is not indexed, add to additional vars
            additional_vars[item.name] = value
        elif item.prefix not in indexed_vars:
            # This is the first item in an indexed var, create a new list
            indexed_vars[item.prefix] = [value]
        else:
            # This is part of an indexed var, append it to the list
            indexed_vars[item.prefix].append(value)

    return any([_evaluate_predicate(predicate, indexed_vars, additional_vars) for predicate in predicates])


def _evaluate_predicate(predicate: str, indexed_vars: Dict[str, Any], additional_vars: Dict[str, Any]) -> bool:
    """Inner predicate evaluator necessary for handling recursive calls on nested quantifiers"""
    # Check if the predicate uses a quantifier:
    if predicate[:6] == 'EXISTS' or predicate[:3] == 'ALL':
        return _handle_quantifier(predicate, indexed_vars, additional_vars)

    # Build the lexer
    lex.lex(module=predicate_lex)

    # Build the parser
    parser = yacc.yacc(module=predicate_yacc)

    # Set the necessary attributes
    parser.i_vars = indexed_vars
    parser.a_vars = additional_vars
    return parser.parse(predicate)


def _handle_quantifier(predicate: str, indexed_vars: Dict[str, Any], additional_vars: Dict[str, Any]) -> bool:
    # The 'iterator' is the variable name immediately following the quantifier
    iterator = predicate.split(' ', 2)[1].replace(':', '')

    # Find the indexed variable being iterated over and its dimension
    sv = predicate.split(f'[{iterator}]', 1)[0].split(' ')[-1]

    # Remove the quantifier and iterator from the beginning of the predicate
    quantified_predicate = predicate.split(': ', 1)[1]

    # The iterator can take values 0 through max_index - 1
    max_index = len(indexed_vars[sv])

    if predicate[:6] == 'EXISTS':
        for i in range(max_index):
            # Add the iterator and its value to the additional_vars
            additional_vars[iterator] = i
            if _evaluate_predicate(quantified_predicate, indexed_vars, additional_vars):
                # There does exist some i
                return True
        return False
    elif predicate[:3] == 'ALL':
        for i in range(max_index):
            # Add the iterator and its value to the additional_vars
            additional_vars[iterator] = i
            if not _evaluate_predicate(quantified_predicate, indexed_vars, additional_vars):
                # There is some i that is not true
                return False
        return True
