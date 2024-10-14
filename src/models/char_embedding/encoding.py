import unicodedata
from typing import Callable


def is_letter(c):
    return unicodedata.category(c).startswith('L')
def is_number(c):
    return unicodedata.category(c).startswith('N')
def is_punctuation(c):
    return unicodedata.category(c).startswith('P')
def is_symbol(c):
    return unicodedata.category(c).startswith('S')
def is_space(c):
    return c.isspace()

DEFAULT_MAPPING = [
    (is_letter, 'A'),
    (is_number, 'N'),
    (is_punctuation, 'P'),
    (is_symbol, 'S'),
    (is_space, ' '),
]

def encode_string(
    s: str,
    mapping: list[tuple[Callable[[str], bool], str]] = DEFAULT_MAPPING
):
    """
    Preprocess a string by replacing characters based on their type.

    Parameters:
    - s: The input string.
    - mapping: A list of tuples (predicate, replacement), where predicate is a function that takes
      a character and returns True or False, and replacement is the string to replace the character with.
      The predicates are checked in order, and the first one that returns True is used.

    Returns:
    - The processed string.
    """
    result = []
    for c in s:
        for predicate, replacement in mapping:
            if predicate(c):
                result.append(replacement)
                break
    return ''.join(result)
