"""Simple noun ambiguity classifier package."""

from .classifier import classify_file, classify_words
from .evaluation import EvaluationReport, evaluate_against_gold

__all__ = [
	"classify_words",
	"classify_file",
	"evaluate_against_gold",
	"EvaluationReport",
]
