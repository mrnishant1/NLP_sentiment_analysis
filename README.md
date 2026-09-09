# Search Engine — MiniBERT Sentiment Analysis

A small, modular NLP project that trains a custom Transformer encoder for Reddit-comment sentiment classification. Despite the repository name, its current implemented capability is **three-class sentiment analysis**, not document retrieval or search.

The workflow first performs masked-language-model (MLM) pretraining on the training comments, then fine-tunes the same encoder with a sentiment classifier. Classes are negative, neutral, and positive.

## What is in the project

| Area | Purpose |
| --- | --- |
| `src/components/dataingestion/` | Validates the CSV and creates a reproducible stratified 80/20 train/test split. |
| `src/components/preprocessing/` | Lowercases text, removes punctuation and English stop words, applies Porter stemming, builds a vocabulary, and pads token IDs. |
| `src/components/model/` | Implements the custom `MiniBERT` encoder, MLM head, sentiment head, and training loops. |
| `src/train_pipeline.py` | End-to-end ingestion, vocabulary creation, MLM pretraining, sentiment fine-tuning, checkpointing, and in-sample evaluation. |
| `src/test_pipeline.py` | Evaluates saved checkpoints on the generated test split. |
| `src/inference.py` | Loads saved artifacts and predicts a sentiment class for text. |
| `config/model_config.yaml` | Model dimensions, training settings, random seed, and all data/artifact paths. |
| `notebooks/BERT_pretrain_finetune.ipynb` | Exploratory notebook for the pretraining/fine-tuning work. |

## Data contract

The configured input file is `raw_data/data.csv`. It must be a CSV with these columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `clean_comment` | text | Comment to classify. |
| `category` | integer | `-1` negative, `0` neutral, `1` positive. |

During ingestion, empty rows, invalid labels, and duplicate text/label pairs are removed. The source file is left unchanged. Labels are converted internally to `0`, `1`, and `2` for PyTorch training.

The included Reddit dataset has 37,249 rows before validation. The current generated split contains 29,285 training rows and 7,322 test rows. Dataset and generated artifacts are intentionally Git-ignored.

## Model and default settings

`MiniBERT` consists of token embeddings, fixed sinusoidal positional encodings, one custom multi-head self-attention encoder block, a feed-forward network, and residual layer normalization. The sentiment model uses the encoded `[CLS]` representation with dropout and a linear three-class head.

Defaults from `config/model_config.yaml`:

- Sequence length: 64 tokens
- Hidden size: 96
- Attention heads: 8
- Feed-forward width: 256
- MLM: 6 epochs, batch size 32, learning rate `1e-4`
- Sentiment fine-tuning: 15 epochs, batch size 32, learning rate `3e-5`

## Setup

Use Python 3.10 or later from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pyyaml
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

`PyYAML` is required by `config/config.py` but is not currently listed in `requirements.txt`, so it is installed explicitly above. Depending on the installed NLTK version, `punkt` may also be required:

```bash
python -c "import nltk; nltk.download('punkt')"
```

The source uses imports from both the repository root (`config`) and `src` (`components`). Run the project with both locations on `PYTHONPATH`:

```bash
export PYTHONPATH=.:src
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH = '.;src'
```

## Run saved-model evaluation and inference

The repository workspace currently contains ignored trained artifacts under `artifacts/`. With them present, evaluate the held-out split:

```bash
PYTHONPATH=.:src python src/test_pipeline.py
```

The saved checkpoint evaluated on the included test split at the time this README was written produced:

```text
Accuracy: 0.3885550396066648
Confusion Matrix (rows=true, cols=pred; class order: negative, neutral, positive)
[[ 270  989  388]
 [ 418 1792  320]
 [ 647 1715  783]]
```

Run the example sentences in the inference script:

```bash
PYTHONPATH=.:src python src/inference.py
```

Or call it from Python:

```python
from inference import inference

prediction = inference("The support team solved my problem quickly.")
# 0 = negative, 1 = neutral, 2 = positive
```

When importing interactively, start Python with `PYTHONPATH=.:src` or add `src` to `sys.path` first.

## Training workflow

1. Put a conforming CSV at `raw_data/data.csv`.
2. Review paths and hyperparameters in `config/model_config.yaml`.
3. Run:

   ```bash
   PYTHONPATH=.:src python src/train_pipeline.py
   ```

4. The pipeline creates `artifacts/validated_data.csv`, `artifacts/train.csv`, `artifacts/test.csv`, `artifacts/vocab.json`, `artifacts/mini_bert_mlm.pth`, and `artifacts/sentiment_model.pth`.

### Current training caveat

As committed, fresh training stops in `train_mlm`: `mask_input_ids` requires a `vocab_size` argument, while `train_mlm` calls it with only the token row. Update that call to pass the vocabulary size before relying on the training command, for example:

```python
m_ids, m_labels = mask_input_ids(row, model.mlm_head.out_features)
```

This documentation does not change model behavior; the existing saved checkpoints can still be evaluated and used for inference as shown above.

## Configuration and artifacts

All paths are relative to the repository root and are configured in `config/model_config.yaml`. CUDA is selected automatically when available; otherwise, the project uses CPU. Checkpoints store the MLM encoder and sentiment classifier separately, and `src/model_loader/load_model.py` reconstructs them for evaluation and inference.

## Limitations

- This is an educational/custom Transformer implementation rather than a pretrained Hugging Face BERT model.
- Its attention implementation does not apply an attention mask, so padding tokens can participate in attention.
- Current test accuracy is modest; use the supplied score as a baseline, not as production-quality validation.
- Data, checkpoints, and vocabulary are ignored by Git, so a fresh clone needs the CSV and must generate or obtain artifacts before inference works.

