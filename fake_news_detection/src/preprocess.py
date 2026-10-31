"""Stage 2 - Pre-processing: lowercase, URL/punctuation/number removal,
stop-word removal, tokenisation and stemming."""
import re
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STEMMER = PorterStemmer()
_STOP = set(ENGLISH_STOP_WORDS)
# ISOT real news start with "CITY (Reuters) -": a known leak, so we strip it.
_AGENCY_PREFIX = re.compile(r"^.{0,60}?\(reuters\)\s*-\s*")


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _STEMMER.stem(word)


def clean_text(text) -> str:
    t = str(text).lower()
    t = _AGENCY_PREFIX.sub("", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)  # drops punctuation + numbers
    return " ".join(_stem(w) for w in t.split() if w not in _STOP and len(w) > 2)
