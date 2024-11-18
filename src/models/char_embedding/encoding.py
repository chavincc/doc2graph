from typing import Callable


def is_alphabet(c):
    return c.isalpha()
def is_digit(c):
    return c.isdigit()
def is_symbol(c):
    return not (c.isalpha() or c.isdigit() or c.isspace())
def is_space(c):
    return c.isspace()

DEFAULT_MAPPING = [
    (is_alphabet, 'A'),
    (is_digit, 'N'),
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
