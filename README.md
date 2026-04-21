# SMS Spam Detection With Unsupervised and Supervised ML

This mini project compares supervised and unsupervised approaches for suspicious SMS detection. It learns normal communication behavior, flags anomalous messages, and then compares that behavior against standard text classifiers.

Unsupervised models:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)

Supervised baselines:

- Logistic Regression
- Multinomial Naive Bayes

The key project novelty is an agreement-based ensemble score:

`agreement_score = (IF + SVM + LOF) / 3`

Messages flagged by multiple models are treated as more reliable anomalies.

## Project Highlights

- Keeps the anomaly detectors label-free while adding supervised baselines for comparison
- Combines `TF-IDF` text signals with handcrafted structural features
- Supports `ham-only` or `full-train` unsupervised training strategies
- Compares supervised, unsupervised, and ensemble performance in one report
- Injects synthetic "new spam" messages to simulate real-world distribution shift
- Exports metrics, predictions, agreement summaries, and case-study examples
- Includes a smoke test with a synthetic SMS dataset

## Folder Structure

```text
.
|-- anomaly_sms/
|   |-- cli.py
|   |-- data.py
|   |-- features.py
|   |-- inference.py
|   |-- models.py
|   |-- pipeline.py
|   `-- reporting.py
|   `-- shift.py
|-- data/
|   `-- raw/
|-- outputs/
|-- streamlit_app.py
|-- tests/
|   `-- test_smoke.py
|-- main.py
`-- requirements.txt
```

## Dataset

Recommended dataset: `SMS Spam Collection Dataset`

Supported layouts:

- `label,message`
- `v1,v2` (common Kaggle export)
- raw UCI-style tab-separated file with `ham<TAB>message`

Place the dataset anywhere on disk and pass it with `--data`.

Example expected rows:

```text
label,message
ham,Hey are we still meeting at 5?
spam,WIN a FREE prize now!!! Click http://spam.example
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py --data "data/raw/SMSSpamCollection" --output-dir outputs
```

Useful options:

```bash
python main.py --data "data/raw/SMSSpamCollection" --train-strategy ham_only --max-features 1500 --test-size 0.3
```

```bash
python main.py --data "data/raw/SMSSpamCollection" --synthetic-spam-count 30 --supervised-spam-fraction 0.5
```

## Demo UI

Launch the lightweight Streamlit demo:

```bash
streamlit run streamlit_app.py
```

What it does:

- trains the project models from your SMS dataset
- lets you paste a new message into one simple input box
- shows a clear risk percentage and risk level
- shows whether the three anomaly models think the message is suspicious
- lists 40 generated suspicious examples and how risky each one looks

Recommended dataset location for easiest launch:

```text
data/raw/SMSSpamCollection
```

## What The Pipeline Does

1. Loads and standardizes the SMS dataset schema
2. Cleans text for TF-IDF
3. Extracts structural features:
   - message length
   - word count
   - digit count
   - URL count
   - special-character count
   - uppercase count
   - uppercase ratio
4. Combines `TF-IDF + structural features`
5. Trains unsupervised models on normal-heavy behavior
6. Trains `LogisticRegression` and `MultinomialNB` on labeled training data
7. Generates an agreement score and ensemble prediction
8. Builds a synthetic "new spam" evaluation set with terms like `UPI`, `crypto`, `OTP fraud`
9. Evaluates each model on:
   - original test data
   - synthetic shifted spam only
   - combined shifted test data
10. Reports:
   - precision
   - recall
   - F1-score
   - accuracy
11. Exports analysis examples showing agreement and disagreement cases

## Outputs

After a run, the project saves:

- `metrics.csv` - performance comparison of each detector and the ensemble
- `predictions.csv` - row-level predictions, scores, and source tags
- `agreement_summary.csv` - vote distribution summary
- `case_studies.csv` - curated examples with heuristic explanations
- `run_config.json` - configuration used for the run

## Suggested Project Story

You can present the project as:

- A lightweight anomaly detector for suspicious SMS messages
- A comparison of supervised and unsupervised detectors
- An agreement-based ensemble that improves reliability over single-model decisions
- A synthetic distribution-shift test showing why anomaly detection matters for unseen spam
- A lightweight real-time UI that demonstrates practical usability
- A practical CPU-friendly approach suitable for classroom and laptop execution

## Expected Insight

In many runs, `LogisticRegression` and `MultinomialNB` will do very well on the original test set because they learn known spam words directly. On the synthetic shifted set, those supervised models may drop when the attack language changes. The unsupervised models often have lower headline accuracy on the standard split, but they can still flag these unseen suspicious messages because the structure and token patterns look abnormal.

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```
