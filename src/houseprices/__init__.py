"""houseprices: reproducible research code for the Kaggle House Prices competition.

Module map (read in this order when studying the project):

- config.py      paths, seeds, cross-validation settings, column groups and mappings
- cleaning.py    the documented semantic cleaning rules (P01 ... C01), applied identically to train and test
- features.py    engineered features and column typing
- preprocess.py  scikit-learn transformers that are FIT INSIDE each training fold
- validation.py  target transform, metric, fold generation, holdout, cross-validation runner
- models.py      model factories and hyperparameter search spaces
- ensemble.py    blending and nested-CV stacking
- registry.py    append-only experiment registry (experiments/registry.jsonl)
- evaluate.py    `python -m houseprices.evaluate --experiment <id>` re-runs a registered CV
- train.py       fits the final system on all training rows and saves artefacts
- predict.py     produces a Kaggle-format submission from the saved artefacts
"""

__version__ = "0.1.0"
