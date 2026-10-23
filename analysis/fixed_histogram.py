"""
Analysis-only patched histogram function.

The production `get_histogram` (src/data/utils.py) and `encode_string`
(src/models/char_embedding/encoding.py) both classify characters using
`str.isalpha()`, which returns False for Unicode combining marks (category
Mn/Mc) -- this misclassifies Thai vowel signs and tone marks (e.g. ิ ี ึ ื ุ ู
ั ่ ้ ๊ ๋ ์) as "symbol" even though they are just how Thai spells vowels,
not punctuation.

This module reimplements get_histogram with a one-line fix so we can measure,
WITHOUT retraining anything, how much of the histogram-vs-BiLSTM gap on
TONDFU is an artifact of this encoding bug vs. a genuine effect of discarding
character order/length.

The trained BiLSTM is intentionally left untouched: it was trained on the
buggy 'A'/'N'/'S' token stream, so re-encoding its input at inference time
would just push it out-of-distribution rather than reflect a real result.
"""
import unicodedata


def is_alphabet_fixed(c: str) -> bool:
    return c.isalpha() or unicodedata.category(c) in ('Mn', 'Mc')


def get_histogram_fixed(contents: list) -> list:
    """Same contract as src.data.utils.get_histogram, but combining marks
    (e.g. Thai vowel signs / tone marks) are folded into the alphabet bucket
    instead of the symbol bucket."""
    c_histograms = []

    for token in contents:
        num_symbols = 0
        num_literals = 0
        num_figures = 0
        num_others = 0

        histogram = [0.0000, 0.0000, 0.0000, 0.0000]

        for symbol in token.replace(" ", ""):
            if is_alphabet_fixed(symbol):
                num_literals += 1
            elif symbol.isdigit():
                num_figures += 1
            else:
                num_others += 1
            num_symbols += 1

        if num_symbols != 0:
            histogram[0] = num_literals / num_symbols
            histogram[1] = num_figures / num_symbols
            histogram[2] = num_others / num_symbols

            if sum(histogram) != 1.0:
                diff = 1.0 - sum(histogram)
                m = max(histogram) + diff
                histogram[histogram.index(max(histogram))] = m

        if histogram[0:3] == [0.0, 0.0, 0.0]:
            histogram[3] = 1.0

        c_histograms.append(histogram)

    return c_histograms
