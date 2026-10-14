"""Train CANYA from scratch on VEP absolute fitness with a fixed train/test split.

Run with the canya package installed in the active Python environment:
    python scripts/vep_only.py
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from canya.modeling import create_model, get_embedded_seqs


SEED = 42
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TARGET = "fitness_mean"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "data/splits/train_sequences.csv"
TEST_PATH = REPO_ROOT / "data/splits/test_sequences.csv"
RUNS_DIR = REPO_ROOT / "runs"
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def load_data(path):
    df = pd.read_csv(path)
    required = {"aa_seq", "sublibrary", TARGET}
    if not required.issubset(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(df.columns))}")
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError(f"{path}: empty dataset or missing required values")

    # Preserve the original construct sequence for tracing saved predictions.
    df["model_sequence"] = df["aa_seq"].map(lambda seq: seq.split("*", 1)[0])
    invalid = df["model_sequence"].map(
        lambda seq: not seq or bool(set(seq) - AMINO_ACIDS) or len(seq) > 20
    )
    if invalid.any():
        raise ValueError(f"{path}: {invalid.sum()} empty or unsupported translated sequences")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
    if not np.isfinite(df[TARGET].to_numpy()).all():
        raise ValueError(f"{path}: nonfinite targets")
    print(f"{path.name}: {len(df)} records; "
          f"{df['aa_seq'].ne(df['model_sequence']).sum()} sequences truncated")
    return df


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)
    if set(train_df["sublibrary"]) & set(test_df["sublibrary"]):
        raise ValueError("Train and test share sublibraries")
    if set(train_df["model_sequence"]) & set(test_df["model_sequence"]):
        raise ValueError("Train and test share peptide sequences after stop-codon truncation")

    x_train = get_embedded_seqs(train_df["model_sequence"].tolist())
    y_train = train_df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    x_test = get_embedded_seqs(test_df["model_sequence"].tolist())
    y_test = test_df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)

    run_name = datetime.now(timezone.utc).strftime("vep_only_%Y%m%dT%H%M%S_%fZ")
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "target": TARGET,
        "initialisation": "random",
        "task": "regression",
        "stop_codon_policy": "truncate_before_first_stop",
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    # Snapshot exact inputs used for this run, including translated sequences.
    train_df.to_csv(output_dir / "train_data.csv", index=False)
    test_df.to_csv(output_dir / "test_data.csv", index=False)

    model = create_model(task="regression")
    model.optimizer.learning_rate.assign(LEARNING_RATE)
    # Fixed training duration: the test set is evaluated only after training.
    history = model.fit(
        x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS
    )
    model.save_weights(str(output_dir / "model.weights.h5"))
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv", index=False)

    metrics = {
        name: float(value)
        for name, value in model.evaluate(x_test, y_test, return_dict=True).items()
    }
    predictions = np.asarray(model.predict(x_test)).reshape(-1)
    test_df["prediction"] = predictions
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # Spearman = Pearson correlation of average ranks; no scipy dependency.
    rho = test_df[TARGET].rank().corr(test_df["prediction"].rank())
    metrics["spearman"] = float(rho) if pd.notna(rho) else None
    # Keras loss also includes regularisation; report prediction-only MSE.
    metrics["mse"] = float(np.mean((y_test.reshape(-1) - predictions) ** 2))
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved run to {output_dir}")


if __name__ == "__main__":
    main()
