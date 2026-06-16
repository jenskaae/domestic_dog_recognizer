# %%

# Several hash types were calculated for each image.
# Cryptographic hashes identify byte-identical files, 
# while perceptual hashes provide a basis for later 
# image-similarity investigations

import hashlib 
from PIL import Image
import imagehash
PHASH_THRESHOLD = 5

manifest_df["sha256_hash"] = None
manifest_df["md5_hash"] = None
manifest_df["a_hash"] = None
manifest_df["d_hash"] = None
manifest_df["p_hash"] = None
manifest_df["duplicate_group_id"] = None
manifest_df["duplicate_group_members"] = None
manifest_df["is_duplicate"] = False
manifest_df["keep"] = True
manifest_df["duplicate_reason"] = None

def calculate_md5(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def calculate_sha256(image_path):
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def compute_a_hash(image_path):
    with Image.open(image_path) as image:
        return imagehash.average_hash(image)
    
def compute_d_hash(image_path):
    with Image.open(image_path) as image:
        return imagehash.dhash(image)
    
def compute_p_hash(image_path):
    with Image.open(image_path) as image:
        return imagehash.phash(image)


# %%
# Each image received multiple hash fingerprints that were 
# stored directly in the manifest.
for index, row in manifest_df.iterrows():
    image_path = PROJECT_ROOT / row[IMAGE_PATH_COLUMN]
    if index % 1000 == 0:
        print(f"Processing image {index:,} of {len(manifest_df):,}")
    try:
        manifest_df.at[index, "sha256_hash"] = calculate_sha256(image_path)
        manifest_df.at[index, "md5_hash"] = calculate_md5(image_path)
        manifest_df.at[index, "a_hash"] = str(compute_a_hash(image_path))
        manifest_df.at[index, "d_hash"] = str(compute_d_hash(image_path))
        manifest_df.at[index, "p_hash"] = str(compute_p_hash(image_path))
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
    
print(manifest_df[["sha256_hash", "md5_hash", "a_hash", "d_hash", "p_hash"]].head())    
print(manifest_df[["sha256_hash", "md5_hash", "a_hash", "d_hash", "p_hash"]].isna().sum())
    
manifest_df.to_csv(MANIFEST_HASHED_PATH, index=False)

# %%
#Before removing duplicates, their extent was measured.
# # this revealed:
# Out of a total of 39,045 images there were:
# 16,874 duplicates
# 22,171 unique images


dup = manifest_df["sha256_hash"].duplicated().sum()

print(f"Duplicate images: {dup}")
print(f"Unique images: {manifest_df['sha256_hash'].nunique()}")


# %%
# Duplicate images were inspected to determine 
# whether they carried identical breed labels.

dup_rows = manifest_df[
    manifest_df["sha256_hash"].duplicated(keep=False)
]

breed_check = (
    dup_rows
    .groupby("sha256_hash")["normalized_breed"]
    .nunique()
)


# %%
# Duplicate images assigned to different breeds were 
# excluded entirely because the correct label could 
# not be determined automatically.

conflict_hashes = breed_counts[
    breed_counts > 1
].index

manifest_df.loc[
    manifest_df["sha256_hash"].isin(conflict_hashes),
    "keep"
] = False

manifest_df.loc[
    manifest_df["sha256_hash"].isin(conflict_hashes),
    "duplicate_reason"
] = "breed_label_conflict"


# %%
# Once conflicts were removed, only a single 
# copy of each exact image was retained.

exact_duplicate_mask = (
    non_conflict_mask
    & manifest_df["sha256_hash"].duplicated(
        keep="first"
    )
)

manifest_df.loc[
    exact_duplicate_mask,
    "keep"
] = False


# %%
# The cleaned manifest became the 
# foundation for dataset splitting and model training.

clean_df = manifest_df[
    manifest_df["keep"]
]

print(len(clean_df))

manifest_df.to_csv(
    MANIFEST_DEDUPED_PATH,
    index=False
)