import os
import typing as _t
from dataclasses import dataclass
from importlib import resources

_USE_WORDFREQ = os.getenv("AMBICLASS_USE_WORDFREQ", "").lower() in {"1", "true", "yes", "on"}

if _USE_WORDFREQ:
    try:  # pragma: no cover - optional dependency
        from wordfreq import zipf_frequency  # type: ignore

        _WORDFREQ_AVAILABLE = True
    except Exception:  # pragma: no cover
        _WORDFREQ_AVAILABLE = False
else:
    _WORDFREQ_AVAILABLE = False

_COMMON_MEMBERSHIP = 5.0
_PROPER_MEMBERSHIP = -4.0
_ZIPF_MULTIPLIER = 0.9
_ZIPF_BIAS = -4.0
_WORD_LENGTH_NEUTRAL = 7
_WORD_LENGTH_MULTIPLIER = 0.04
_HAS_COMMON_SUFFIX = 0.0
_CAPITALIZATION_PENALTY = 0.0
_CAPITALIZATION_ZIPF_LOW = 2.0
_CAPITALIZATION_ZIPF_RANGE = 4.0
_COMMON_THRESHOLD = 0.05
_NOT_THRESHOLD = -40.0

@dataclass(frozen=True)
class ClassificationResult:
    word: str
    label: str
    score: float
    reason: str


_common_cache: set[str] | None = None
_common_lower_cache: set[str] | None = None
_proper_cache: set[str] | None = None
_proper_lower_cache: set[str] | None = None


def _load_common_words() -> set[str]:
    global _common_cache
    if _common_cache is None:
        txt = resources.files(__package__).joinpath("data/common.txt")
        with txt.open("r", encoding="utf-8") as handle:
            _common_cache = {line.strip() for line in handle if line.strip()}
    return _common_cache


def _load_common_lower() -> set[str]:
    global _common_lower_cache
    if _common_lower_cache is None:
        _common_lower_cache = {w.lower() for w in _load_common_words()}
    return _common_lower_cache


def _load_proper_words() -> set[str]:
    global _proper_cache
    if _proper_cache is None:
        try:
            txt = resources.files(__package__).joinpath("data/proper.txt")
        except FileNotFoundError:  # pragma: no cover - defensive
            txt = None
        members: set[str] = set()
        if txt is not None and getattr(txt, "is_file", lambda: False)():
            with txt.open("r", encoding="utf-8") as handle:
                members = {line.strip() for line in handle if line.strip()}
        _proper_cache = members
    return _proper_cache


def _load_proper_lower() -> set[str]:
    global _proper_lower_cache
    if _proper_lower_cache is None:
        _proper_lower_cache = {w.lower() for w in _load_proper_words()}
    return _proper_lower_cache


def _zipf(word: str) -> float:
    if not _WORDFREQ_AVAILABLE:
        return 0.0
    return zipf_frequency(word, "en")


def classify_word(word: str) -> ClassificationResult:
    word = word.strip()
    if not word:
        return ClassificationResult(word, "unknown", 0.0, "empty")

    lower = word.lower()
    commons = _load_common_words()
    commons_lower = _load_common_lower()
    propers = _load_proper_words()
    propers_lower = _load_proper_lower()

    score = 0.0
    if word in commons or (word.islower() and lower in commons_lower):
        score += _COMMON_MEMBERSHIP
    if propers and word in propers:
        score += _PROPER_MEMBERSHIP
    elif propers_lower and word.islower() and lower in propers_lower:
        score += _PROPER_MEMBERSHIP

    zipf = _zipf(lower)
    score += _ZIPF_MULTIPLIER * (zipf + _ZIPF_BIAS)

    length_delta = len(word) - _WORD_LENGTH_NEUTRAL
    score -= _WORD_LENGTH_MULTIPLIER * length_delta

    if _CAPITALIZATION_PENALTY and word and word[0].isupper():
        freq_factor = 0.0
        if _WORDFREQ_AVAILABLE:
            freq_factor = (zipf - _CAPITALIZATION_ZIPF_LOW) / _CAPITALIZATION_ZIPF_RANGE
            freq_factor = max(0.0, min(1.0, freq_factor))
        score -= _CAPITALIZATION_PENALTY * freq_factor


    label = "likely ambiguous" if score >= _COMMON_THRESHOLD else "likely non-ambiguous"
    if score <= _NOT_THRESHOLD:
        label = "likely non-ambiguous"

    reason_bits = [f"zipf={zipf:.2f}" if _WORDFREQ_AVAILABLE else "zipf=n/a", f"lenΔ={length_delta}"]
    reason = ";".join(reason_bits)
    return ClassificationResult(word, label, score, reason)


def classify_words(words: _t.Iterable[str]) -> tuple[list[str], list[str]]:
    ambiguous: list[str] = []
    proper: list[str] = []
    commons = _load_common_words()
    commons_lower = _load_common_lower()
    propers = _load_proper_words()
    propers_lower = _load_proper_lower()

    for raw in words:
        res = classify_word(raw)
        if res.word == "" or res.label == "unknown":
            continue
        lower = res.word.lower()
        treat_as_common = (
            res.label == "likely ambiguous"
            or res.word in commons
            or (res.word.islower() and lower in commons_lower)
        )
        treat_as_proper = (
            res.word in propers
            or (res.word.islower() and lower in propers_lower)
        )
        if treat_as_common and not treat_as_proper:
            ambiguous.append(res.word)
        else:
            proper.append(res.word)
    return ambiguous, proper


def classify_file(input_path: str, ambiguous_output: str, proper_output: str) -> None:
    with open(input_path, "r", encoding="utf-8") as reader:
        words = [line.strip() for line in reader if line.strip()]
    ambiguous, proper = classify_words(words)
    with open(ambiguous_output, "w", encoding="utf-8") as a_out:
        for word in sorted(dict.fromkeys(ambiguous)):
            a_out.write(f"{word}\n")
    with open(proper_output, "w", encoding="utf-8") as p_out:
        for word in sorted(dict.fromkeys(proper)):
            p_out.write(f"{word}\n")
