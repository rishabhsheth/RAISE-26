# RAISE-26

A small utility project to compute sentiment for news titles in Dataset B and produce an enriched CSV with sentiment labels and scores.

**Highlights**
- **Simple CLI**: runs sentiment analysis over the `title` column and writes an updated CSV.
- **Resilient backends**: tries Transformers (fast GPU/CPU model), falls back to VADER or TextBlob when needed.
- **Portable**: uses `pathlib` for robust path handling and supports a `config/default_inputs.json` file for default inputs.

**Contents**
- **data/**: input dataset(s) (example: `dataset_B_news_subset_3500.csv`).
- **src/title_sentiment_analysis.py**: main script that computes sentiment and writes output.
- **requirements.txt**: pinned packages used for development and running the script.

**Quick Start**

1. Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

2. Create the `data/` folder and add the provided dataset

If the repository doesn't already contain a `data/` folder, create it and copy the provided dataset into it. The example dataset filename is `dataset_B_news_subset_3500.csv`.

```bash
# create the folder (if missing) and copy the dataset file
mkdir -p data
# On Windows (PowerShell)
Copy-Item ..\path\to\dataset_B_news_subset_3500.csv data\
# On macOS / Linux
cp /path/to/dataset_B_news_subset_3500.csv data/
```

3. Run the sentiment script (uses defaults if no args provided):

```bash
# Run with default inputs from config/default_inputs.json (or repo root default_inputs.json)
python src/title_sentiment_analysis.py

# Or pass explicit input and output paths
python src/title_sentiment_analysis.py data/dataset_B_news_subset_3500.csv updated_dataset_b.csv
```

**Configuration (optional)**

Place a `default_inputs.json` under `config/default_inputs.json` or at the repo root with keys for `in_csv` and `out_csv`.

Example `config/default_inputs.json`:

```json
{
	"in_csv": "data/dataset_B_news_subset_3500.csv",
	"out_csv": "data/updated_dataset_b.csv"
}
```

**What the script produces**
- Adds two columns to the dataset: `title_sentiment` (POSITIVE / NEGATIVE / NEUTRAL) and `title_sentiment_score` (numeric score).
- Writes the enriched CSV to the configured output path (default `data/updated_dataset_b.csv`).

**Notes & Tips**
- If `transformers` + `torch` are installed, the script will attempt to load a distilled BERT model (`distilbert-base-uncased-finetuned-sst-2-english`) which performs well for short text sentiment.
- For machines without GPUs or where Transformers fails, the code falls back to NLTK VADER or TextBlob automatically.
- `requirements.txt` contains versions used during development; you may adjust versions to suit your environment.

**Troubleshooting**
- If you see import errors, ensure your virtual environment is activated and `pip install -r requirements.txt` completed successfully.
- If transformers model download fails behind a proxy, pre-download the model or run on a machine with internet access.
