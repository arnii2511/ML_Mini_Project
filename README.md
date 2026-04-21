# SMS Scam and Spam Detection Using Classical ML

This mini project detects suspicious SMS messages using a combination of:

- supervised learning for known spam patterns
- unsupervised learning for new or unusual suspicious patterns

The project is built around a practical idea:

`A user only wants to know whether a message looks like scam/spam or not.`

So the final system combines both types of models into one scam-focused decision.

## Models Used

Supervised models:

- Logistic Regression
- Multinomial Naive Bayes

Unsupervised models:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)

## Core Idea

The project uses two internal signals:

1. `Known Spam Match`
   This comes from the supervised models.
   It helps catch common spam or scam messages that look similar to patterns already present in the dataset.

2. `New Suspicious Pattern`
   This comes from the unsupervised models.
   It helps catch unusual or previously unseen scam styles.

Final user-facing result:

- `Scam / Spam Risk`

The final risk is designed so that common spam is not treated as safe just because the unsupervised models did not flag it.

## Why Both Supervised and Unsupervised?

Supervised models are very strong when the spam pattern is already known.

Unsupervised models are useful because in real life:

- new scam formats appear often
- wording changes frequently
- not every new attack style exists in training data

So the project story is:

- supervised models catch known spam well
- unsupervised models help when a suspicious message follows a newer pattern

## Project Highlights

- Uses `TF-IDF + handcrafted features`
- Supports `ham_only` or `all` for unsupervised training
- Includes supervised and unsupervised model comparison
- Includes synthetic suspicious-message generation for stress testing
- Provides a lightweight Streamlit UI for live demo
- Runs on CPU only
- Includes unit tests

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
|   |-- reporting.py
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

Recommended dataset:

- `SMS Spam Collection Dataset`

Supported layouts:

- `label,message`
- `v1,v2`
- raw tab-separated format like `ham<TAB>message`

Recommended local path:

```text
data/raw/SMSSpamCollection
```

Example:

```text
label,message
ham,Hey are we still meeting at 5?
spam,WIN a FREE prize now!!! Click http://spam.example
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run The Offline Pipeline

```bash
python main.py --data "data/raw/SMSSpamCollection" --output-dir outputs
```

Useful examples:

```bash
python main.py --data "data/raw/SMSSpamCollection" --train-strategy ham_only --max-features 1500 --test-size 0.3
```

```bash
python main.py --data "data/raw/SMSSpamCollection" --synthetic-spam-count 30 --supervised-spam-fraction 0.5
```

## Run The Demo UI

```bash
streamlit run streamlit_app.py
```

The Streamlit app:

- trains the models from the SMS dataset
- lets the user paste a new SMS
- shows `Scam / Spam Risk`
- shows `Known Spam Match`
- shows `New Suspicious Pattern`
- gives a short human-readable reason
- lists 40 generated suspicious examples with their risk levels

## How The System Works

### 1. Data Loading

The dataset loader:

- reads CSV, TSV, or TXT input
- finds label and text columns automatically
- standardizes them to:
  - `label`
  - `text`
  - `is_anomaly`

`ham -> 0`

`spam -> 1`

### 2. Text Cleaning

The text is cleaned before feature extraction:

- lowercase conversion
- URL replacement with a token
- punctuation cleanup
- whitespace cleanup

### 3. Feature Engineering

The project uses two feature groups.

Text features:

- TF-IDF
- up to `1500` features by default
- unigrams and bigrams

Structural features:

- message length
- word count
- digit count
- URL count
- special character count
- uppercase count
- uppercase ratio

Final feature vector:

- `TF-IDF + structural features`

### 4. Unsupervised Training

The anomaly models are:

- Isolation Forest
- One-Class SVM
- LOF

These models do not learn direct `spam` labels.

Instead, they learn which messages look normal or familiar.

Two modes are supported:

- `ham_only`
  - anomaly models train only on normal messages
- `all`
  - anomaly models train on all training messages

Important:

- `ham_only` applies only to the unsupervised models
- supervised models are trained separately

### 5. Supervised Training

The supervised models are:

- Logistic Regression
- Multinomial Naive Bayes

These use labels directly and learn known spam patterns.

The spam amount used in supervised training can be reduced with:

- `supervised_spam_fraction`

This helps simulate limited exposure to known spam examples.

### 6. Final Decision Logic

The app internally computes:

- `Known Spam Match`
  - average supervised spam confidence
- `New Suspicious Pattern`
  - unsupervised agreement score

Then it creates the final user-facing decision:

- `Scam / Spam Risk`

Current practical rule:

- common known spam should still be flagged even if unsupervised models do not consider it unusual
- suspicious new-looking messages can also be flagged even when they do not perfectly match known spam wording

## Evaluation

The project compares:

- supervised models
- unsupervised models
- the unsupervised ensemble

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score

The offline pipeline also supports synthetic suspicious-message evaluation to simulate distribution shift.

## Outputs

After running the offline pipeline, the project saves:

- `metrics.csv`
- `predictions.csv`
- `agreement_summary.csv`
- `case_studies.csv`
- `run_config.json`

## Suggested Report Story

You can explain the project like this:

- Supervised learning performs best on known spam patterns.
- Unsupervised learning is weaker on standard labeled benchmarks but useful for suspicious new patterns.
- A practical scam detector should not rely on only one of them.
- So this system combines both ideas into a single scam/spam risk output.

## Current Demo Message

For a normal user, the app should be explained simply:

- paste an SMS
- see if it looks like scam/spam
- get a risk score and short reason

The user does not need to understand anomaly detection terminology.

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py"
```
