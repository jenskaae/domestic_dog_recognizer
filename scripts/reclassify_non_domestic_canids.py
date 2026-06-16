# %%
# Before recategorization the dataset contained 38,078 dog images and 967 not_dog images.

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DEDUPED_PATH = PROJECT_ROOT / "data/intermittent/manifest_deduped.csv"
VALIDATION_DIAGNOSTICS_RECAT_PATH = PROJECT_ROOT / "data/intermittent/validation_diagnostics_recat.csv"
manifest_df = pd.read_csv(MANIFEST_DEDUPED_PATH)
clean_df = manifest_df[manifest_df["keep"] == True].copy()

print("Before deduplication:")
print(manifest_df["class_label"].value_counts())

print("\nAfter deduplication, keep=True:")
print(clean_df["class_label"].value_counts())

# %%
# The dog datasets contained dholes, dingoes and 
# African hunting dogs labelled as dog.

targets = [
    "dhole",
    "dingo",
    "african"
]

print(
    clean_df[
        clean_df["normalized_breed"]
            .str.lower()
            .str.contains("|".join(targets), na=False)
    ]["normalized_breed"]
        .value_counts()
)


# %%
# Several of the most common false negatives
# involved canids.

diagnostics = pd.read_csv(
    PROJECT_ROOT / "data/intermittent/validation_diagnostics.csv"
)

mistakes = diagnostics[
    diagnostics["actual"] != diagnostics["predicted"]
]

print(
    mistakes.groupby("normalized_breed")
            .size()
            .sort_values(ascending=False)
            .head(20)
)


# %%
# 942 images were recategorized.

MANIFEST_RECAT_CANIDS_PATH = PROJECT_ROOT / "data/intermittent/manifest_recat_canids.csv"

before = pd.read_csv(MANIFEST_DEDUPED_PATH)
after = pd.read_csv(MANIFEST_RECAT_CANIDS_PATH)

changed = (
    before["class_label"] != after["class_label"]
).sum()

print(changed) # 942!

after_clean = after[after["keep"] == True].copy()

print("After recategorization, all rows:")
print(after["class_label"].value_counts())

print("\nAfter recategorization, keep=True:")
print(after_clean["class_label"].value_counts())

changed_clean = (
    (before["class_label"] != after["class_label"])
    & (before["keep"] == True)
).sum()

print("Rows recategorized, all rows:", changed)
print("Rows recategorized, keep=True:", changed_clean)

print("\nReclassified canids remaining after deduplication:")
print(
    after_clean[
        after_clean["normalized_breed"].isin(
            ["african_hunting_dog", "dingo", "dhole"]
        )
    ][["normalized_breed", "class_label"]]
    .value_counts()
)
# %%
# After recategorization the dataset contained
# 37,136 dog images and 1,909 not_dog images.

print(after["class_label"].value_counts())

# %%
# Show the false negatives sorted by confidence

VALIDATION_MISTAKES_PATH = PROJECT_ROOT / "data/intermittent/validation_mistakes.csv"
false_negatives = pd.read_csv(VALIDATION_MISTAKES_PATH)

false_negatives.sort_values(
    "confidence",
    ascending=False
)[[
    "normalized_breed",
    "confidence"
]].head(20)

# %%
# Do the recategorization

recat_breeds = [
    "african_hunting_dog",
    "dingo",
    "dhole",
]

mask = (
    manifest_df["keep"]
    & manifest_df["normalized_breed"].isin(recat_breeds)
)

manifest_df.loc[mask, "class_name"] = "not_dog"
manifest_df.loc[mask, "class_label"] = 0

manifest_df.to_csv(MANIFEST_RECAT_CANIDS_PATH, index=False)