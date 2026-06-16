# %%
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_INPUT_PATH = PROJECT_ROOT / "data" / "intermittent" / "validation_diagnostics.csv"
MISTAKES_OUTPUT_PATH = PROJECT_ROOT / "data" / "intermittent" / "validation_mistakes.csv"

diagnostics_df = pd.read_csv(PROJECT_ROOT / DIAGNOSTICS_INPUT_PATH)


false_negatives = diagnostics_df[
    (diagnostics_df["actual"] == 1)
    & (diagnostics_df["predicted"] == 0)
].copy()

false_negatives = false_negatives.sort_values(
    "confidence",
    ascending=False
)

print(
    false_negatives[
        ["normalized_breed", "confidence"]
    ].head(20)
)
# %%
false_positives = diagnostics_df[
    (diagnostics_df["actual"] == 0)
    & (diagnostics_df["predicted"] == 1)
].copy()

false_positives = false_positives.sort_values(
    "confidence",
    ascending=False
)

print(
    false_positives[
        ["raw_parent_folder", "confidence"]
    ].head(20)
)

# %%
print(
    false_negatives["normalized_breed"]
    .value_counts()
    .head(20)
)


# %%
print(
    false_positives["raw_parent_folder"]
    .value_counts()
    .head(20)
)