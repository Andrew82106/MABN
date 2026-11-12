"""Small evidence-overlap features using retrieved text only, never annotations."""
from collections import Counter
import re
import unicodedata
import numpy as np

NAMES = ['source_word_present', 'source_word_log_frequency', 'source_ngram_support',
         'numeric_word', 'numeric_word_unsupported', 'capitalized_word_unsupported']
WORD = re.compile(r'[^\W_]+(?:[.,][0-9]+)?', re.UNICODE)


def normalized(word):
    return unicodedata.normalize('NFKC', word).casefold()


def features(response, retrieved_text, offsets):
    """Character-weighted word features on the unchanged generator's BPE grid.

    Source matches are cues, not correctness labels. A word occurring somewhere
    in a document does not prove the whole claim, and a paraphrase can be valid.
    """
    words = list(WORD.finditer(response))
    source = [normalized(m.group()) for m in WORD.finditer(retrieved_text)]
    target = [normalized(m.group()) for m in words]
    counts = Counter(source)
    ngrams = {n: {tuple(source[i:i+n]) for i in range(len(source)-n+1)} for n in range(2, 5)}
    per_word = np.zeros((len(words), len(NAMES)), np.float32)
    for i, match in enumerate(words):
        word = match.group(); norm = target[i]
        present = bool(counts[norm])
        largest = int(present)
        for n in range(2, 5):
            for first in range(max(0, i-n+1), min(i+1, len(words)-n+1)):
                if tuple(target[first:first+n]) in ngrams[n]: largest = n
        numeric = any(ch.isnumeric() for ch in word)
        capitalized = len(word) > 1 and word[0].isupper()
        per_word[i] = [present, np.log1p(min(counts[norm], 50)), largest / 4,
                       numeric, numeric and not present, capitalized and not present]
    result = np.zeros((len(offsets), len(NAMES)), np.float32)
    for i, (a, b) in enumerate(offsets):
        assert 0 <= a <= b <= len(response)
        numerator, denominator = np.zeros(len(NAMES), np.float64), 0
        for j, match in enumerate(words):
            lo, hi = max(a, match.start()), min(b, match.end())
            overlap = sum(ch.isalnum() for ch in response[lo:hi]) if lo < hi else 0
            numerator += overlap * per_word[j]
            denominator += overlap
        if denominator: result[i] = numerator / denominator
    assert np.isfinite(result).all()
    return result


def selfcheck():
    response = 'Ada wrote in 1843, NOT 1901.'
    offsets = [[0, 1], [1, 3], [3, 9], [13, 15], [15, 17], [18, 21], [22, 26]]
    x = features(response, 'Ada wrote in 1843.', offsets)
    assert np.all(x[:2, 0] == 1) and np.all(x[3:5, 4] == 0)
    assert x[-1, 3] == 1 and x[-1, 4] == 1
    assert x[5, 5] == 1
    # The question is deliberately absent from the API: unsupported numbers
    # supplied only in a question cannot masquerade as retrieved evidence.
    y = features('Émile ２０２０', 'émile 2020', [[0, 5], [6, 10]])
    assert np.all(y[:, 0] == 1) and np.all(y[:, 4] == 0)
    print('EVIDENCE_FEATURES_CPU_CHECK_PASSED')


if __name__ == '__main__': selfcheck()
