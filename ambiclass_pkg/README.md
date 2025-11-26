
# ambiguity-simple

A minimal noun ambiguity classifier that ships with a curated common noun list and tuned heuristic
parameters. It separates a word list into two files:

* **Likely ambiguous** – words that behave like common nouns
* **Likely non-ambiguous** – words that behave like proper nouns

The bundled weights were selected from the `ambiguity_lexicons` experiments (best accuracy ≈ 81%).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install ./ambiclass_pkg
```

Classify a list (one token per line):

```bash
pip install -e ./ambiclass_pkg

# classify a list
ambiguity-simple words.txt ambiguous.txt proper.txt
```

Project layout conventions:

- Place wordlists to classify under `data/input/`.
- Keep gold-standard TSVs (with `Word` and `Truth` columns) in `data/gold/`.
- Write classifier outputs and evaluation artefacts to `data/output/`.
	See `data/README.md` for a quick reminder.

Evaluate existing outputs against a gold standard (tab-separated `Word` / `Truth` columns):

```bash
ambiguity-simple evaluate data/gold/gold_standard.tsv data/output/ambiguous.txt \
	data/output/proper.txt --show-mismatches 10 --json-report data/output/eval_summary.json
```

The JSON report is optional; omit `--json-report` if you only need console metrics.

### Optional `wordfreq` support

The classifier defaults to a word-frequency-free heuristic for higher precision. To activate
`wordfreq` scoring, set `AMBICLASS_USE_WORDFREQ=1` (or `true/yes/on`) when running the CLI or using
the library.

```bash
AMBICLASS_USE_WORDFREQ=1 ambiguity-simple words.txt ambiguous.txt proper.txt
```

## Library usage

```python
from ambiguity_simple import classify_words

ambiguous, proper = classify_words(["London", "happiness", "AI"])
print(ambiguous)  # ['happiness']
print(proper)     # ['London', 'AI']
```

## Implementation details

* Uses `wordfreq.zipf_frequency` when `AMBICLASS_USE_WORDFREQ` is enabled (falls back to 0
	otherwise).
* Membership boosts: +4 if the word is in the bundled common list.
* Frequency weighting: `0.85 * (zipf + -4.5)`
* Length deviation: subtract `0.03 * (len(word) - 6)`
* Common threshold: ≥ 0.1 → *likely ambiguous*; ≤ -80 → *likely non-ambiguous*.
* Suffix bonus currently set to zero (placeholder for future tweaks).

## Data provenance

Richard's unvetted wordlist is the current default wordlist.


Before releasing:

- Update the version in `ambiclass_pkg/pyproject.toml`.
- Regenerate `data/output/` artefacts via the CLI if you ship sample outputs.
- Run `ambiguity-simple evaluate ...` against your latest gold standard to confirm metrics.
