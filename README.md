# catalog-tfm

Shallow Keras transformer on ingested earthquake CSVs: predicts **next-event magnitude** and **inter-event time Δt** (seconds). Training uses **chronological** train/val/test **per catalog**; optional **test-only** CSVs (filename keywords) are evaluated only on the test split. Metrics are reported on **continuous** values and on **discretized** bins (magnitude edges and Δt in fixed-width second bins).

## Setup

```bash
cd /path/to/transformer
python -m venv venvs/catalog_tfm
source venvs/catalog_tfm/bin/activate   # Windows: venvs\catalog_tfm\Scripts\activate
pip install -r requirements.txt
pip install --no-deps -e ../eq_mag_prediction
pip install -e .
```

The virtualenv lives under `venvs/catalog_tfm` so its directory name is `catalog_tfm` without colliding with the import package folder `catalog_tfm/`.

## Train

```bash
catalog-tfm-train --data-dir /path/to/ingested --epochs 20
# omit CSVs whose basename contains a substring (repeat for several keywords):
catalog-tfm-train --exclude-filename-keyword major --exclude-filename-keyword nz
# CSVs matching these keywords are test-only (not used in train/val):
catalog-tfm-train --test-only-keyword jma --dt-bin-seconds 1200 --mag-step 0.02
```

Default `--data-dir` resolves to `../eq_mag_prediction/results/catalogs/ingested` from the current working directory.

## Notebook

Open `scripts/train_and_visualize.ipynb`.

## Repository

Remote: https://github.com/Ner-Ber/transformer-catalog
