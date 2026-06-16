# %%
# At this time we have a manifest file created from previous steps. 
# This file had normalized breed names in the normalized_breed column, 
# and deduplicated the data by allowing a fileter keep = true. The 
# resulting manifest was saved to manifest_split.csv by 
# clean_df = manifest_df[manifest_df["keep"] == True].copy()
# clean_df.to_csv(MANIFEST_SPLIT_PATH, index=False)

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SPLIT_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_split.csv"

check_df = pd.read_csv(MANIFEST_SPLIT_PATH)

train_manifest = check_df[check_df["split"] == "train"]
val_manifest = check_df[check_df["split"] == "val"]
test_manifest = check_df[check_df["split"] == "test"]

print(f"Train set size: {len(train_manifest)}")
print(f"Validation set size: {len(val_manifest)}")
print(f"Test set size: {len(test_manifest)}")

# Train set size: 15298
# Validation set size: 3262
# Test set size: 3263

# %%
# We have a class imbalance in the training set, 
# with many more dog images than not-dog images.
# we also have many breeds with only a few images. 

train_dogs = train_manifest[
    train_manifest["class_label"] == 1
].copy()

train_not_dogs = train_manifest[
    train_manifest["class_label"] == 0
].copy()

print(f"Original number of dog images: {len(train_dogs)}")
print(f"Original number of not-dog images: {len(train_not_dogs)}")
# Original number of dog images: 14289
# Original number of not-dog images: 1009


# %%
# To create a more balanced training set,
# we will limit the number of dog images per breed. 
# We do this by saying max 7 per breed

MAX_DOGS_PER_BREED = 7
RANDOM_SEED = 42
sampled_parts = []
for breed, group in train_dogs.groupby("normalized_breed"):
    sampled_group = group.sample(
        n=min(MAX_DOGS_PER_BREED, len(group)),
        random_state=RANDOM_SEED
    )
    sampled_parts.append(sampled_group)
    print(f"Breed: {breed}, Original: {len(group)}, Sampled: {len(sampled_group)}")
print(len(sampled_parts))

# %%
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_TRAIN_REDUCED_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_train_reduced.csv"
MANIFEST_VAL_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_val.csv"
MANIFEST_TEST_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_test.csv"
train_dogs_sampled = pd.concat(sampled_parts, ignore_index=True)

train_df_reduced = pd.concat(
    [train_dogs_sampled, train_not_dogs],
    ignore_index=True
)

train_df_reduced = train_df_reduced.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

train_df_reduced.to_csv(MANIFEST_TRAIN_REDUCED_PATH, index=False)
val_manifest.to_csv(MANIFEST_VAL_PATH, index=False)
test_manifest.to_csv(MANIFEST_TEST_PATH, index=False)

print("Saved:")
print(MANIFEST_TRAIN_REDUCED_PATH.resolve())
print(MANIFEST_VAL_PATH.resolve())
print(MANIFEST_TEST_PATH.resolve())


# %%

print("Original train:")
print(train_manifest["class_label"].value_counts())

print("\nReduced train:")
print(train_df_reduced["class_label"].value_counts())

# %%
# Check dog / not-dog distribution in each split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SPLIT_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_split.csv"

check_df = pd.read_csv(MANIFEST_SPLIT_PATH)

split_class_counts = (
    check_df
    .groupby(["split", "class_label"])
    .size()
    .unstack(fill_value=0)
)

split_class_counts = split_class_counts.rename(
    columns={
        0: "not_dog",
        1: "dog"
    }
)

print(split_class_counts)
print()

split_class_percentages = (
    split_class_counts
    .div(split_class_counts.sum(axis=1), axis=0)
    .round(3)
)

print(split_class_percentages)

# %%
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_SPLIT_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_split.csv"
MANIFEST_TRAIN_REDUCED_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_train_reduced.csv"
MANIFEST_VAL_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_val.csv"
MANIFEST_TEST_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_test.csv"

MAX_DOGS_PER_BREED = 7
RANDOM_SEED = 42

manifest = pd.read_csv(MANIFEST_SPLIT_PATH)

train_manifest = manifest[manifest["split"] == "train"].copy()
val_manifest = manifest[manifest["split"] == "val"].copy()
test_manifest = manifest[manifest["split"] == "test"].copy()

train_dogs = train_manifest[train_manifest["class_label"] == 1].copy()
train_not_dogs = train_manifest[train_manifest["class_label"] == 0].copy()

print("Original train:")
print(train_manifest["class_label"].value_counts().rename(index={0: "not_dog", 1: "dog"}))


# %%
sampled_parts = []

for breed, group in train_dogs.groupby("normalized_breed"):
    sampled_group = group.sample(
        n=min(MAX_DOGS_PER_BREED, len(group)),
        random_state=RANDOM_SEED
    )
    sampled_parts.append(sampled_group)

train_dogs_sampled = pd.concat(sampled_parts, ignore_index=True)

train_df_reduced = pd.concat(
    [train_dogs_sampled, train_not_dogs],
    ignore_index=True
)

train_df_reduced = train_df_reduced.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

print("Reduced train:")
print(train_df_reduced["class_label"].value_counts().rename(index={0: "not_dog", 1: "dog"}))

print("\nDog breeds sampled:")
print(train_dogs_sampled["normalized_breed"].nunique())

print("\nMax dog images per breed:")
print(train_dogs_sampled["normalized_breed"].value_counts().max())


# %%
train_df_reduced.to_csv(MANIFEST_TRAIN_REDUCED_PATH, index=False)
val_manifest.to_csv(MANIFEST_VAL_PATH, index=False)
test_manifest.to_csv(MANIFEST_TEST_PATH, index=False)

print("Saved:")
print(MANIFEST_TRAIN_REDUCED_PATH.resolve())
print(MANIFEST_VAL_PATH.resolve())
print(MANIFEST_TEST_PATH.resolve())
# %%
