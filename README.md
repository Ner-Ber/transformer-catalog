# catalog-tfm

Shallow Keras transformer trained on ingested earthquake CSVs (see `eq_mag_prediction` `results/catalogs/ingested`).

## Setup

```bash
cd /path/to/transformer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps -e ../eq_mag_prediction
pip install -e .
```

## Train

```bash
catalog-tfm-train --data-dir /path/to/ingested --epochs 20
```

Default `--data-dir` resolves to `../eq_mag_prediction/results/catalogs/ingested` from the current working directory.

## Notebook

Open `scripts/train_and_visualize.ipynb`.
