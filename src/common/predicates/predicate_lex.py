"""
Tokenizer for evaluating predicate expression strings
@author Dallan Goldblatt
"""

# List of token names
tokens = (
    'NUMBER',
    'VAR',
    'LPAREN',
    'RPAREN',
    'LBRACK',
    'RBRACK',
    'COMMA',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'AND',
    'OR',
    'NOT',
    'COMPARISON',
    'TRUTH'
)

# Dict of reserved words and their types
reserved = {
    'AND': 'AND',
    'OR': 'OR',
    'NOT': 'NOT',
    'TRUE': 'TRUTH',
    'FALSE': 'TRUTH'
 }

# Regular expression rules for simple tokens
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACK = r'\['
t_RBRACK = r'\]'
t_COMMA = r'\,'
t_PLUS = r'\+'
t_MINUS = r'\-'
t_TIMES = r'\*'
t_DIVIDE = r'\/'
t_COMPARISON = r'[!=<>]+'


# Convert reserved words to their special type, otherwise keep token as a VAR
def t_VAR(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = reserved.get(t.value, 'VAR')
    return t


# Convert number tokens from strings to their values
def t_NUMBER(t):
    r'[+-]?\d+(?:\.\d+)?'
    value = float(t.value)
    t.value = int(value) if value - round(value) == 0 else value
    return t


# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'


# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
