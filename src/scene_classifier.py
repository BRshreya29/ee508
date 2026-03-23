"""Stage A4 — Scene Classifier (6-class MLP on CLIP CLS tokens).

Reference: doc 05 — Scene Classifier.
- 2-layer MLP: Linear(512, 128) → GELU → Dropout(0.2) → Linear(128, 6)
- Trained offline on Places365 subset before main training run
- At inference: frozen, produces [6] softmax probability per frame
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

SCENE_CLASSES = ["kitchen", "street", "office", "gym", "living_room", "outdoor"]
NUM_SCENE_CLASSES = len(SCENE_CLASSES)

# Places365 → working scene class mapping
PLACES365_TO_SCENE = {
    "kitchen": "kitchen",
    "dining_room": "kitchen",
    "street": "street",
    "road": "street",
    "highway": "street",
    "office": "office",
    "conference_room": "office",
    "gym": "gym",
    "fitness_center": "gym",
    "living_room": "living_room",
    "bedroom": "living_room",
    "forest": "outdoor",
    "park": "outdoor",
    "field": "outdoor",
    "mountain": "outdoor",
    "garden": "outdoor",
}


class SceneClassifier(nn.Module):
    """6-class scene classifier on CLIP CLS tokens."""

    def __init__(self, input_dim=512, num_classes=NUM_SCENE_CLASSES):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, clip_cls_token):
        """
        clip_cls_token: [B, 512]
        Returns: [B, 6] raw logits
        """
        return self.mlp(clip_cls_token)


class Places365CLIPDataset(Dataset):
    """Dataset of CLIP CLS tokens extracted from Places365 images."""

    def __init__(self, features_dir):
        """
        features_dir: directory containing per-class subdirectories with .pt files
        """
        self.samples = []
        scene_to_idx = {s: i for i, s in enumerate(SCENE_CLASSES)}

        for class_name in SCENE_CLASSES:
            class_dir = os.path.join(features_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith(".pt"):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, scene_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = torch.load(path, weights_only=True)  # [512]
        return feat, label


def train_scene_classifier(
    features_dir,
    checkpoint_path="checkpoints/scene_mlp.pt",
    epochs=20,
    lr=1e-3,
    batch_size=32,
    val_split=0.2,
    device="cpu",
):
    """Train the scene classifier MLP on Places365 CLIP features.

    Args:
        features_dir: directory with per-class CLIP feature .pt files
        checkpoint_path: where to save the trained model
        epochs: number of training epochs
        lr: learning rate
        batch_size: training batch size
        val_split: fraction for validation
        device: torch device string
    """
    device = torch.device(device)

    # Load dataset
    dataset = Places365CLIPDataset(features_dir)
    if len(dataset) == 0:
        print("WARNING: No training data found. Saving random weights.")
        model = SceneClassifier()
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        return model

    # Split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Model + optimizer
    model = SceneClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)

        train_loss /= len(train_set)

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch + 1}/{epochs} — loss: {train_loss:.4f}, val_acc: {val_acc:.4f}")

    print(f"Best val accuracy: {best_acc:.4f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    return model
