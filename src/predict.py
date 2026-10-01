"""Predict five probabilities from persons and situations, without reading responses."""
import features  # Sets thread limits before importing numerical libraries.
import argparse
import joblib
import pandas as pd
from features import CLASSES, join_inputs
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("results/model/model.joblib"))
    parser.add_argument("--persons", type=Path, required=True)
    parser.add_argument("--situations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    frame = join_inputs(pd.read_parquet(args.persons), pd.read_parquet(args.situations))
    # Joblib uses pickle: load only artifacts from a trusted source.
    model = joblib.load(args.model)
    if model.classes != CLASSES:
        raise ValueError("Model class order differs from public output contract")
    probabilities = model.predict_proba(frame)
    output = frame[["situation_id", "person_id"]].reset_index(drop=True)
    for index, name in enumerate(CLASSES):
        output[name] = probabilities[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
