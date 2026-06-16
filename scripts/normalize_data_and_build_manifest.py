# %%
# Breed normalization starts from the manifest, 
# not from the image folders themselves.
from pathlib import Path
import pandas as pd
import re

MANIFEST_PATH = Path("../../data/intermittent/manifest_aug_2.csv")

manifest = pd.read_csv(MANIFEST_PATH)

# %%
# Before normalizing anything, determine what names actually exist.
breed_names = sorted(
    manifest["raw_breed_name"]
    .dropna()
    .unique()
)

print(f"Unique breed names: {len(breed_names)}")
breed_names[:20]


# %%
# Similar names, fuzzy compare, were automatically identified 
# as normalization candidates.
from difflib import SequenceMatcher

candidate_pairs = []

for i in range(len(breed_names)):
    for j in range(i + 1, len(breed_names)):
        score = SequenceMatcher(
            None,
            breed_names[i],
            breed_names[j]
        ).ratio()

        if score >= 0.78:
            candidate_pairs.append({
                "breed_a": breed_names[i],
                "breed_b": breed_names[j],
                "score": score
            })

candidate_df = (
    pd.DataFrame(candidate_pairs)
    .sort_values("score", ascending=False)
)

candidate_df.head(20)




# %%
# After reviewing the candidate pairs and making modifications 
# to the "normalized_breed" column in the "breed_candidates.csv"
# file, we load the modified CSV back into a DataFrame. We then 
# create a mapping from the original breed names (breed_a and breed_b)
# to the normalized breed name. 
# Finally, we print out the mapping for verification.
reviewed_df = pd.read_csv("breed_candidates_modified.csv")

breed_map = {}

for _, row in reviewed_df.iterrows():

    if pd.notna(row["normalized_breed"]):

        breed_map[row["breed_a"]] = row["normalized_breed"]
        breed_map[row["breed_b"]] = row["normalized_breed"]

for breed in breed_map :
    print(f"{breed} -> {breed_map[breed]}")
    
# %%

# %%
# Idea revealed:
# Simple string similarity was insufficient for cases such as:
# mexicanhairless
# mexican_hairless
# setting similarity threshold too low would result in many false positives, 
# while setting it too high would miss these cases.

# Therefore a token-based comparison was added:

# This section defines utility functions for tokenizing breed names and calculating similarity scores
# based on tokens. The tokenize function splits a breed name into tokens using underscores, hyphens, or spaces 
# as delimiters. The token_similarity function calculates the similarity between two tokens using SequenceMatcher. 
# The token_coverage_score function calculates the average of the best token similarity scores for each token 
# in one list compared to the tokens in another list. These functions are used later to identify candidate 
# breed name pairs that may have similar tokens even if their overall string similarity is not high.
def tokenize(name):
    return re.split(r"[_\-\s]+", name)

def token_similarity(token_a, token_b):
    return SequenceMatcher(None, token_a, token_b).ratio()

def token_coverage_score(tokens_a, tokens_b):
    """
    For each token in tokens_a, find the best matching token in tokens_b.
    Return the average of those best scores.
    """
    best_scores = []

    for token_a in tokens_a:
        best_score = max(
            token_similarity(token_a, token_b)
            for token_b in tokens_b
        )
        best_scores.append(best_score)

    return sum(best_scores) / len(best_scores)    
    
# %%
# This section uses the token-based similarity approach to identify candidate breed name pairs. 
# We iterate through all unique breed names and compare each pair using both the whole string similarity 
# and the token coverage score. 
# If either the whole string similarity exceeds 0.78 or the best token coverage score exceeds 0.95, 
# we consider the pair as a candidate for normalization and store it in a list. 
# Finally, we convert the list of candidate pairs into a DataFrame and sort it by the best token coverage 
# and whole string similarity for review.
breed_names = sorted(manifest["raw_breed_name"].dropna().unique())
candidate_rows = []

for i in range(len(breed_names)):
    for j in range(i + 1, len(breed_names)):
        breed_a = breed_names[i]
        breed_b = breed_names[j]

        tokens_a = tokenize(breed_a)
        tokens_b = tokenize(breed_b)

        whole_score = SequenceMatcher(None, breed_a, breed_b).ratio()

        coverage_a_to_b = token_coverage_score(tokens_a, tokens_b)
        coverage_b_to_a = token_coverage_score(tokens_b, tokens_a)

        best_token_coverage = max(coverage_a_to_b, coverage_b_to_a)

        if whole_score >= 0.78 or best_token_coverage >= 0.95:
            candidate_rows.append({
                "breed_a": breed_a,
                "breed_b": breed_b,
                "whole_score": whole_score,
                "coverage_a_to_b": coverage_a_to_b,
                "coverage_b_to_a": coverage_b_to_a,
                "best_token_coverage": best_token_coverage,
            })

candidate_df = pd.DataFrame(candidate_rows).sort_values(
    ["best_token_coverage", "whole_score"],
    ascending=False
)

candidate_df

# %%
# Normalization decisions were not made automatically. 
# Candidate pairs were reviewed manually.
candidate_df["normalized_breed"] = None

candidate_df.to_csv(
    "breed_candidates.csv",
    index=False
)




# %%
# this cell is for adding the normalized_breed_names to the manifest based on the mapping we created 
# from the reviewed candidate pairs. 
# We create a new column "normalized_breed" in the manifest by replacing the raw breed names 
# using the breed_map.
# Finally, we print out the unique normalized breed names for verification.
# All downstream processing uses one normalized breed column.
manifest["normalized_breed"] = (
    manifest["raw_breed_name"]
    .replace(breed_map)
)

for breed in manifest["normalized_breed"].unique():
    print(breed) 


# %%
# The result was inspected before being persisted for later pipeline stages.
check_df = (
    manifest[
        ["raw_breed_name",
         "normalized_breed"]
    ]
    .drop_duplicates()
    .sort_values(
        ["normalized_breed",
         "raw_breed_name"]
    )
)

check_df

manifest.to_csv(
    "../../data/intermittent/manifest_breeds_normalized_v2.csv",
    index=False
)