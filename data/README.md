# Data Directory

Use this directory to keep the files needed for the ambiguity classifier together.

- Put the word list you plan to classify in `data/input/` (one token per line, UTF-8 encoded).
- Store the reference gold standard TSV, such as `gold_standard.tsv`, in `data/gold/`.
- Direct the classifier outputs (`ambiguous.txt`, `proper.txt`, or similar) to `data/output/`.

The evaluation routine expects the gold standard file to have `Word` and `Truth` columns separated by tabs. The classifier and evaluator do not create these subfolders automatically, so create them as needed before running the commands.
