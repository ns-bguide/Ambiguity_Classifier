"""Evaluation routines for the ambiguity classifier."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

Label = str


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics describing how predictions line up with the gold standard."""

    total_words: int
    total_common: int
    total_proper: int
    predicted_common: int
    predicted_proper: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float
    precision: float | None
    recall: float | None
    f1: float | None
    missing_words: tuple[str, ...]
    false_positive_words: tuple[str, ...]
    false_negative_words: tuple[str, ...]
    extra_common_predictions: tuple[str, ...]
    extra_proper_predictions: tuple[str, ...]

    def as_dict(self, include_details: bool = True) -> dict[str, object]:
        """Return a serialisable view of the report."""

        summary: dict[str, object] = {
            "total_words": self.total_words,
            "total_common": self.total_common,
            "total_proper": self.total_proper,
            "predicted_common": self.predicted_common,
            "predicted_proper": self.predicted_proper,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "missing_words_count": len(self.missing_words),
            "extra_common_predictions_count": len(self.extra_common_predictions),
            "extra_proper_predictions_count": len(self.extra_proper_predictions),
        }
        if include_details:
            summary["missing_words"] = list(self.missing_words)
            summary["false_positive_words"] = list(self.false_positive_words)
            summary["false_negative_words"] = list(self.false_negative_words)
            summary["extra_common_predictions"] = list(self.extra_common_predictions)
            summary["extra_proper_predictions"] = list(self.extra_proper_predictions)
        return summary


def evaluate_against_gold(
    gold_standard_path: str | Path,
    ambiguous_predictions_path: str | Path,
    proper_predictions_path: str | Path,
) -> EvaluationReport:
    """Compare classifier outputs against a gold standard TSV file."""

    gold_labels = _load_gold_labels(gold_standard_path)
    gold_words = set(gold_labels)
    gold_common = {word for word, label in gold_labels.items() if label == "common"}
    gold_proper = gold_words - gold_common

    predicted_common_all = _load_prediction_words(ambiguous_predictions_path)
    predicted_proper_all = _load_prediction_words(proper_predictions_path)

    extra_common_predictions = predicted_common_all - gold_words
    extra_proper_predictions = predicted_proper_all - gold_words

    predicted_common = predicted_common_all & gold_words
    predicted_proper = predicted_proper_all & gold_words

    overlapping_predictions = predicted_common & predicted_proper
    if overlapping_predictions:
        overlap_preview = ", ".join(sorted(overlapping_predictions)[:5])
        raise ValueError(
            f"Found {len(overlapping_predictions)} words present in both prediction files; "
            f"examples: {overlap_preview}."
        )

    missing_words = gold_words - (predicted_common | predicted_proper)

    true_positive_words = gold_common & predicted_common
    false_negative_words = gold_common - predicted_common
    false_positive_words = predicted_common & gold_proper
    true_negative_words = gold_proper & predicted_proper

    total_words = len(gold_words)
    total_common = len(gold_common)
    total_proper = len(gold_proper)
    predicted_common_count = len(predicted_common)
    predicted_proper_count = len(predicted_proper)
    true_positive = len(true_positive_words)
    true_negative = len(true_negative_words)
    false_positive = len(false_positive_words)
    false_negative = len(false_negative_words)

    accuracy = (true_positive + true_negative) / total_words if total_words else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )

    return EvaluationReport(
        total_words=total_words,
        total_common=total_common,
        total_proper=total_proper,
        predicted_common=predicted_common_count,
        predicted_proper=predicted_proper_count,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        missing_words=_sort_words(missing_words),
        false_positive_words=_sort_words(false_positive_words),
        false_negative_words=_sort_words(false_negative_words),
        extra_common_predictions=_sort_words(extra_common_predictions),
        extra_proper_predictions=_sort_words(extra_proper_predictions),
    )


def _load_gold_labels(path: str | Path) -> dict[str, Label]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or {"Word", "Truth"} - set(reader.fieldnames):
            raise ValueError(
                "Gold standard file must contain 'Word' and 'Truth' columns separated by tabs."
            )
        labels: dict[str, Label] = {}
        for row in reader:
            word = (row.get("Word") or "").strip()
            if not word:
                continue
            truth_raw = (row.get("Truth") or "").strip()
            truth = _normalise_truth(truth_raw)
            existing = labels.get(word)
            if existing is not None and existing != truth:
                raise ValueError(
                    f"Conflicting labels detected for word '{word}': '{existing}' vs '{truth}'."
                )
            labels[word] = truth
    return labels


def _load_prediction_words(path: str | Path) -> set[str]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _normalise_truth(label: str) -> Label:
    lowered = label.strip().lower()
    if lowered in {"common", "ambiguous", "ambiguous noun"}:
        return "common"
    if lowered in {"proper", "proper noun"}:
        return "proper"
    raise ValueError(f"Unrecognised label '{label}'. Expected 'Common' or 'Proper'.")


def _sort_words(words: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(words, key=lambda item: (item.lower(), item)))
