from pathlib import Path

import pandas as pd

BASE = Path("/data/input/preprocessed")

sources = [
    ("MELD_BONN_hemisphere.csv",     "hemisphere_text"),
    ("MELD_BONN_lobe.csv",           "lobe_text"),
    ("MELD_BONN_hemisphere_lobe.csv", "hemisphere_lobe_text"),
]

KEY_COLS = ["DATA_PATH", "ROI_PATH"]


def load_and_prepare(filename: str, new_text_col: str) -> pd.DataFrame:
    """Load a single-mode CSV and rename its text column to new_text_col."""
    path = BASE / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    missing_keys = [k for k in KEY_COLS if k not in df.columns]
    if missing_keys:
        raise ValueError(f"{filename}: missing required key columns {missing_keys}")

    candidate_cols = [c for c in df.columns if c not in KEY_COLS and c != 'aal']

    if not candidate_cols:
        df[new_text_col] = pd.NA
        return df[KEY_COLS + [new_text_col]]

    if len(candidate_cols) > 1:
        raise ValueError(
            f"{filename}: multiple non-key columns found {candidate_cols}. "
            "Please specify which one to use."
        )

    original_text_col = candidate_cols[0]
    df = df[KEY_COLS + [original_text_col]].copy()
    df = df.rename(columns={original_text_col: new_text_col})
    df = df.drop_duplicates(subset=KEY_COLS)
    return df


if __name__ == "__main__":
    merged = None

    for fname, new_col in sources:
        try:
            part = load_and_prepare(fname, new_col)
            merged = part if merged is None else merged.merge(part, on=KEY_COLS, how="outer")
            print(f"Added: {fname} -> {new_col}, shape={part.shape}")
        except Exception as e:
            print(f"Skipped {fname}: {e}")

    if merged is None:
        raise RuntimeError("Failed to assemble any input files.")

    merged["no_text"] = "full brain"

    out_path = BASE / "MELD_BONN_mixed.csv"
    merged.to_csv(out_path, index=False)

    print(f"\nSaved to: {out_path} (shape={merged.shape})")
    print("Columns:", list(merged.columns))
