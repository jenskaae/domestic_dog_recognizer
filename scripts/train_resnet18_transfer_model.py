# %%
from pathlib import Path
import time

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


# %%
# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_TRAIN_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_train_reduced.csv"
MANIFEST_VAL_PATH = PROJECT_ROOT / "data" / "intermittent" / "manifest_val.csv"

MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "resnet18_dog_not_dog.pt"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 42


# %%
# Dataset class

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

        return image, label


# %%
# Validation helper

def calculate_accuracy(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


# %%
# Load manifests

train_df = pd.read_csv(MANIFEST_TRAIN_PATH)
val_df = pd.read_csv(MANIFEST_VAL_PATH)

print("Train rows:", len(train_df))
print("Validation rows:", len(val_df))

print("\nTrain labels:")
print(train_df["class_label"].value_counts())

print("\nValidation labels:")
print(val_df["class_label"].value_counts())


# %%
# Transforms and dataloaders

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

train_dataset = DogManifestDataset(train_df, PROJECT_ROOT, transform)
val_dataset = DogManifestDataset(val_df, PROJECT_ROOT, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# %%
# Model setup

torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

model = resnet18(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

number_of_features = model.fc.in_features
model.fc = nn.Linear(number_of_features, 2)

model = model.to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# %%
# Training loop

run_start = time.perf_counter()

for epoch in range(NUM_EPOCHS):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_accuracy = calculate_accuracy(model, train_loader, device)
    val_accuracy = calculate_accuracy(model, val_loader, device)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} "
        f"| Loss: {running_loss:.4f} "
        f"| Train acc: {train_accuracy:.3f} "
        f"| Val acc: {val_accuracy:.3f}"
    )

# Epoch 1/5 | Loss: 24.5929 | Train acc: 0.941 | Val acc: 0.922
# Epoch 2/5 | Loss: 12.3305 | Train acc: 0.960 | Val acc: 0.956
# Epoch 3/5 | Loss: 11.0143 | Train acc: 0.962 | Val acc: 0.966
# Epoch 4/5 | Loss: 9.1537 | Train acc: 0.969 | Val acc: 0.967
# Epoch 5/5 | Loss: 8.4085 | Train acc: 0.969 | Val acc: 0.975

# %%
# Save trained model

MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

run_end = time.perf_counter()

print("\nSaved model to:", MODEL_OUTPUT_PATH.resolve())
print(f"Total training time: {run_end - run_start:.1f}s")

# %%
import sys
print(sys.executable)
