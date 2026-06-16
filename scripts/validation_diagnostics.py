# %%
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_VAL_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_val.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_dog_not_dog.pt"

DIAGNOSTICS_OUTPUT_PATH = PROJECT_ROOT / "data" / "intermittent" / "validation_diagnostics.csv"
MISTAKES_OUTPUT_PATH = PROJECT_ROOT / "data" / "intermittent" / "validation_mistakes.csv"

BATCH_SIZE = 32


class DogManifestDataset(Dataset):
    def __init__(self, manifest_df, project_root, transform=None):
        self.manifest_df = manifest_df.reset_index(drop=True)
        self.project_root = project_root
        self.transform = transform

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, index):
        row = self.manifest_df.iloc[index]

        image_path = self.project_root / row["image_path"]
        label = int(row["class_label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, row["image_path"]


def run_validation(model, dataloader, device):
    model.eval()

    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, image_paths in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidence_values, predictions = probabilities.max(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for image_path, actual, predicted, confidence in zip(
                image_paths,
                labels.cpu(),
                predictions.cpu(),
                confidence_values.cpu()
            ):
                results.append({
                    "image_path": image_path,
                    "actual": int(actual),
                    "predicted": int(predicted),
                    "confidence": float(confidence),
                    "correct": int(actual) == int(predicted)
                })

    results_df = pd.DataFrame(results)
    accuracy = correct / total

    return accuracy, results_df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    val_manifest = pd.read_csv(MANIFEST_VAL_PATH)

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    val_dataset = DogManifestDataset(
        manifest_df=val_manifest,
        project_root=PROJECT_ROOT,
        transform=transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    number_of_features = model.fc.in_features
    model.fc = nn.Linear(number_of_features, 2)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model = model.to(device)

    accuracy, val_results_df = run_validation(model, val_loader, device)

    diagnostics_df = val_results_df.merge(
        val_manifest,
        on="image_path",
        how="left"
    )

    mistakes_df = diagnostics_df[
        diagnostics_df["correct"] == False
    ].copy()

    diagnostics_df.to_csv(DIAGNOSTICS_OUTPUT_PATH, index=False)
    mistakes_df.to_csv(MISTAKES_OUTPUT_PATH, index=False)

    print(f"Validation accuracy: {accuracy:.3f}")
    print(f"Validation rows: {len(diagnostics_df)}")
    print(f"Mistakes: {len(mistakes_df)}")
    print("Saved diagnostics:", DIAGNOSTICS_OUTPUT_PATH.resolve())
    print("Saved mistakes:", MISTAKES_OUTPUT_PATH.resolve())

    print("\nMistakes by actual/predicted:")
    print(pd.crosstab(mistakes_df["actual"], mistakes_df["predicted"]))

    if "normalized_breed" in mistakes_df.columns:
        print("\nTop mistake breeds:")
        print(
            mistakes_df["normalized_breed"]
            .value_counts()
            .head(20)
        )


if __name__ == "__main__":
    main()
 
# %%
diagnostics_df = pd.read_csv(DIAGNOSTICS_OUTPUT_PATH)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

precision = precision_score(
    diagnostics_df["actual"],
    diagnostics_df["predicted"]
)

recall = recall_score(
    diagnostics_df["actual"],
    diagnostics_df["predicted"]
)

f1 = f1_score(
    diagnostics_df["actual"],
    diagnostics_df["predicted"]
)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 score:  {f1:.3f}")
# %%
