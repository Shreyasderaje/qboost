"""Adapters that map public datasets onto the QBoost feature contract.

The federated pipeline, the quantum models and the privacy stack are
agnostic to the data source.  To leave simulation mode:

- diabetes  → ``load_pima_csv`` on the PIMA Indians Diabetes CSV
- tb        → export NIH ChestX-ray14 (or a local X-ray folder) with
              ``load_image_folder``; recommend pre-computed CNN embeddings
              for production quality
- malaria   → the Kaggle "Malaria Cell Images" dataset with
              ``load_image_folder`` (positives/negatives sub-folders)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# PIMA column order (standard Kaggle CSV) mapped onto the QBoost schema.
_PIMA_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
_PIMA_TO_QBOOST = ["glucose", "bmi", "age", "insulin", "blood_pressure", "pregnancies", "skinfold", "pedigree"]
_ZERO_AS_MISSING = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}


def load_pima_csv(path: str | Path) -> dict:
    """Load the PIMA Indians Diabetes dataset into the diabetes feature schema."""
    import pandas as pd

    frame = pd.read_csv(path)
    missing = [c for c in _PIMA_COLUMNS + ["Outcome"] if c not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing expected PIMA columns: {missing}")

    for col in _ZERO_AS_MISSING:
        frame[col] = frame[col].replace(0, np.nan)
        frame[col] = frame[col].fillna(frame[col].median())

    frame = frame.rename(
        columns=dict(zip(_PIMA_COLUMNS, ["pregnancies", "glucose", "blood_pressure", "skinfold", "insulin", "bmi", "pedigree", "age"]))
    )
    X = frame[_PIMA_TO_QBOOST].to_numpy(dtype=np.float64)
    y = frame["Outcome"].to_numpy(dtype=np.int64)
    return {"X": X, "y": y, "feature_names": _PIMA_TO_QBOOST}


def _image_features(img_path: Path, size: int = 48) -> np.ndarray:
    """A compact, dependency-light image descriptor.

    16-bin intensity histogram + 12-block gradient energy + 4 shape stats.
    For production, replace with a frozen CNN embedding (e.g. a mobile net
    trained on the target task) — the rest of the pipeline is unchanged.
    """
    from PIL import Image

    img = Image.open(img_path).convert("L").resize((size, size))
    a = np.asarray(img, dtype=np.float64) / 255.0

    hist, _ = np.histogram(a, bins=16, range=(0.0, 1.0))
    hist = hist / max(1, hist.sum())

    gx = np.diff(a, axis=1)
    gy = np.diff(a, axis=0)
    energy = (gx[:-1, :-1] ** 2 + gy[:-1, :-1] ** 2).ravel()
    blocks = np.array_split(energy, 12)
    block_energy = np.array([b.mean() if b.size else 0.0 for b in blocks])

    stats = np.array([a.mean(), a.std(), np.percentile(a, 10), np.percentile(a, 90)])
    return np.concatenate([hist, block_energy, stats])


def load_image_folder(root: str | Path, limit_per_class: int | None = None, seed: int = 0) -> dict:
    """Load a folder of ``<class-name>/<image>`` pairs into feature space.

    Class labels are 0/1 by sorted class-name order (e.g. ``negatives``,
    ``positives`` or ``Uninfected``, ``Parasitized``).
    """
    root = Path(root)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if len(class_dirs) != 2:
        raise ValueError(f"expected exactly 2 class sub-folders, found {[d.name for d in class_dirs]}")

    rng = np.random.default_rng(seed)
    features, labels = [], []
    for label, d in enumerate(class_dirs):
        files = sorted(p for p in d.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        if limit_per_class and len(files) > limit_per_class:
            files = list(rng.choice(files, limit_per_class, replace=False))
        for f in files:
            features.append(_image_features(f))
            labels.append(label)

    names = [f"hist_{i}" for i in range(16)] + [f"grad_{i}" for i in range(12)] + ["mean", "std", "p10", "p90"]
    return {"X": np.vstack(features), "y": np.asarray(labels, dtype=np.int64), "feature_names": names}
