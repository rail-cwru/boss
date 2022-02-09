"""
Semantics interpreter for evaluating predicate expression strings
@author Dallan Goldblatt
"""

# Get the token map from the lexer.
from common.predicates.predicate_lex import tokens

# Get the arbitrarily defined functions for predicates
from common.predicates.functions import dispatcher

def p_expression_or(p):
    'expression : expression OR aexpression'
    p[0] = p[1] or p[3]

def p_expression_aexpression(p):
    'expression : aexpression'
    p[0] = p[1]

def p_aexpression_and(p):
    'aexpression : aexpression AND nexpression'
    p[0] = p[1] and p[3]

def p_aexpression_nexpression(p):
    'aexpression : nexpression'
    p[0] = p[1]

def p_nexpression_not(p):
    'nexpression : NOT term'
    p[0] = not p[2]

def p_nexpression_term(p):
    'nexpression : term'
    p[0] = p[1]

def p_term_paren(p):
    'term : LPAREN expression RPAREN'
    p[0] = p[2]

def p_term_truth(p):
    'term : TRUTH'
    p[0] = p[1] == 'TRUE'

def p_term_comparison(p):
    'term : calculation COMPARISON calculation'
    if p[2] == '==':
        p[0] = p[1] == p[3]
    elif p[2] == '!=':
        p[0] = p[1] != p[3]
    elif p[2] == '<':
        p[0] = p[1] < p[3]
    elif p[2] == '<=':
        p[0] = p[1] <= p[3]
    elif p[2] == '>':
        p[0] = p[1] > p[3]
    elif p[2] == '>=':
        p[0] = p[1] >= p[3]

def p_calculation_summand(p):
    'calculation : summand'
    p[0] = p[1]

def p_summand_plus(p):
    'summand : summand PLUS factor'
    p[0] = p[1] + p[3]

def p_summand_minus(p):
    'summand : summand MINUS factor'
    p[0] = p[1] - p[3]

def p_summand_factor(p):
    'summand : factor'
    p[0] = p[1]

def p_factor_times(p):
    'factor : factor TIMES value'
    p[0] = p[1] * p[3]

def p_factor_divide(p):
    'factor : factor DIVIDE value'
    p[0] = p[1] / p[3]

def p_factor_value(p):
    'factor : value'
    p[0] = p[1]

def p_value_paren(p):
    'value : LPAREN calculation RPAREN'
    p[0] = p[2]

def p_value_func(p):
    'value : VAR LPAREN param_list RPAREN'
    p[0] = dispatcher[p[1]](*p[3])

def p_param_list_param(p):
    'param_list : param'
    p[0] = [p[1]]

def p_param_list_param_list(p):
    'param_list : param_list COMMA param'
    p[1].append(p[3])
    p[0] = p[1]

def p_param_calculation(p):
    'param : calculation'
    p[0] = p[1]

def p_value_vari(p):
    'value : VAR index'
    p[0] = p.parser.i_vars[p[1]][p[2]]

def p_value_var(p):
    'value : VAR'
    try:
        p[0] = p.parser.a_vars[p[1]]
    except KeyError:
        p[0] = p.parser.i_vars[p[1]]

def p_value_num(p):
    'value : NUMBER'
    p[0] = p[1]

def p_index_var(p):
    'index : LBRACK VAR RBRACK'
    p[0] = p.parser.a_vars[p[2]]

def p_index_num(p):
    'index : LBRACK NUMBER RBRACK'
    p[0] = p[2]

# Error rule for syntax errors
def p_error(p):
    print(p)
    print("Syntax error in input!")