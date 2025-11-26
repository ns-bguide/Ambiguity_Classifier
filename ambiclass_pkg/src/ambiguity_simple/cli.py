import argparse
import json
import sys
from pathlib import Path

from .classifier import classify_file
from .evaluation import EvaluationReport, evaluate_against_gold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ambiguity-simple",
        description="Split a word list into likely ambiguous (common nouns) and likely non-ambiguous (proper nouns).",
    )
    parser.add_argument("input", help="Input file with one word per line.")
    parser.add_argument("ambiguous_output", help="Output file for likely ambiguous/common nouns.")
    parser.add_argument("proper_output", help="Output file for likely non-ambiguous/proper nouns.")
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ambiguity-simple evaluate",
        description="Evaluate classifier outputs against a gold standard TSV file.",
    )
    parser.add_argument("gold_standard", help="Gold standard TSV with Word and Truth columns.")
    parser.add_argument(
        "ambiguous_predictions",
        help="File containing the classifier's ambiguous/common noun predictions.",
    )
    parser.add_argument(
        "proper_predictions",
        help="File containing the classifier's proper noun predictions.",
    )
    parser.add_argument(
        "--json-report",
        help="Optional path to write a JSON summary of the evaluation.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only include aggregate counts in the JSON report (omit detailed word lists).",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=0,
        metavar="N",
        help="Display up to N false positives and false negatives in the console output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "evaluate":
        eval_parser = build_evaluate_parser()
        eval_args = eval_parser.parse_args(argv[1:])
        report = evaluate_against_gold(
            eval_args.gold_standard,
            eval_args.ambiguous_predictions,
            eval_args.proper_predictions,
        )
        _print_report(report, eval_args.show_mismatches)
        if eval_args.json_report:
            _write_json_report(
                eval_args.json_report,
                report.as_dict(include_details=not eval_args.summary_only),
            )
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    classify_file(args.input, args.ambiguous_output, args.proper_output)
    return 0


def _print_report(report: EvaluationReport, mismatch_limit: int) -> None:
    precision = f"{report.precision:.4f}" if report.precision is not None else "n/a"
    recall = f"{report.recall:.4f}" if report.recall is not None else "n/a"
    f1 = f"{report.f1:.4f}" if report.f1 is not None else "n/a"

    print("Evaluation summary")
    print(f"Total words: {report.total_words}")
    print(f"Gold common: {report.total_common}")
    print(f"Gold proper: {report.total_proper}")
    print(f"Predicted common: {report.predicted_common}")
    print(f"Predicted proper: {report.predicted_proper}")
    print(f"Accuracy: {report.accuracy:.4f}")
    print(f"Precision (common): {precision}")
    print(f"Recall (common): {recall}")
    print(f"F1 (common): {f1}")
    print(f"True positives: {report.true_positive}")
    print(f"True negatives: {report.true_negative}")
    print(f"False positives: {report.false_positive}")
    print(f"False negatives: {report.false_negative}")
    print(f"Missing from predictions: {len(report.missing_words)}")
    print(f"Extra common predictions: {len(report.extra_common_predictions)}")
    print(f"Extra proper predictions: {len(report.extra_proper_predictions)}")

    if mismatch_limit > 0:
        _print_examples("False positives", report.false_positive_words, mismatch_limit)
        _print_examples("False negatives", report.false_negative_words, mismatch_limit)
        _print_examples("Missing in outputs", report.missing_words, mismatch_limit)


def _print_examples(title: str, words: tuple[str, ...], limit: int) -> None:
    if not words:
        return
    shown = ", ".join(words[:limit])
    more = "" if len(words) <= limit else f" (and {len(words) - limit} more)"
    print(f"{title}: {shown}{more}")


def _write_json_report(path: str, data: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
