# BanglaIndoorVQA: code

Code accompanying the Data in Brief article **"<PAPER TITLE>"** (<AUTHORS>, <YEAR>).
Dataset: <DATASET DOI / URL>. Archived code release: <ZENODO DOI>.

This repository contains the scripts used to (1) build the released dataset and
(2) run the zero-shot baseline. It does not contain the images or raw household data.

## Repository layout

```
common/                 shared text cleaning and strict/relaxed match functions
  normalize.py
dataset_construction/   building and verifying the released dataset
  prepare_dataset.py
  merge_annotations.py
  make_stats.py
  check_dataset.py
baseline/               zero-shot baseline run
  00_prep_images.py
  01_build_items.py
  02_run_model.py
data/README.md          where to get the dataset and the expected layout
```

## Setup

```bash
git clone https://github.com/<USERNAME>/<REPO>.git
cd <REPO>
pip install -r requirements.txt
```

Tested with Python <VERSION> on <Google Colab / GPU model>.

## Data

Download the dataset from <DATASET DOI / URL> and place it as described in
[`data/README.md`](data/README.md).

## Run order

Run all commands from the repository root.

### A. Dataset construction

| Step | Command | Input | Output |
|---|---|---|---|
| 1 | `python dataset_construction/prepare_dataset.py <args>` | raw images | processed images + manifest |
| 2 | `python dataset_construction/merge_annotations.py <args>` | per-household exports | merged annotation file |
| 3 | `python dataset_construction/make_stats.py <args>` | merged file | descriptive statistics |
| 4 | `python dataset_construction/check_dataset.py <args>` | released package | verification report |

### B. Zero-shot baseline

| Step | Command | Input | Output |
|---|---|---|---|
| 1 | `python baseline/00_prep_images.py <args>` | images | images resized to 768 px (short side) |
| 2 | `python baseline/01_build_items.py <args>` | dataset | `items.jsonl` (seed-42 shuffle) |
| 3 | `python baseline/02_run_model.py <args>` | `items.jsonl`, images | model predictions |

Replace `<args>` with the exact arguments each script accepts (`--help` lists them).

## Settings used in the paper

- Model: <MODEL NAME AND VERSION>
- Random seed: 42
- Generation settings: <temperature, max new tokens, batch size>
- Hardware: <GPU / runtime>

## License

Code: MIT (see `LICENSE`). The dataset is licensed separately, see <DATASET DOI / URL>.

## Citation

If you use this code, please cite the article and the archived release:
<CITATION>
