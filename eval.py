import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score
import json
import argparse

from src.model import SceneActivityModel
from src.dataset import ActionGenomeFeatures
from src.output_heads import compute_mAP

def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model on Test set")
    parser.add_argument("--features_dir", type=str, default="features")
    parser.add_argument("--labels_file", type=str, default="data/labels.json")
    parser.add_argument("--split_file", type=str, default="data/split.json")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load the test split
    with open(args.split_file, "r") as f:
        split_data = json.load(f)
    test_vids = split_data.get("test", [])
    
    # 2. Setup Dataset & Loader
    test_dataset = ActionGenomeFeatures(
        features_dir=args.features_dir,
        labels_file=args.labels_file,
        video_ids=test_vids
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 3. Load Model
    model = SceneActivityModel().to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded {args.checkpoint} (val_loss={ckpt.get('val_loss', 'N/A')})")
    else:
        print("ERROR: Checkpoint not found.")
        return
    model.eval()

    # 4. Evaluation Loop
    all_act_logits = []
    all_act_targets = []
    
    print(f"Evaluating on {len(test_vids)} test videos...")
    
    with torch.no_grad():
        for batch in test_loader:
            clip, diff, obj, scene, obj_mask, targets = (
                batch["clip"].to(device),
                batch["diff"].to(device),
                batch["objects"].to(device),
                batch["scene"].to(device),
                batch["obj_mask"].to(device),
                batch["targets"]
            )
            act_targets = targets["activity"].to(device)

            act_logits, loc_preds, sg_logits, _ = model(clip, diff, obj, scene, obj_mask)
            
            all_act_logits.append(act_logits.cpu())
            all_act_targets.append(act_targets.cpu())

    all_act_logits = torch.cat(all_act_logits, dim=0)
    all_act_targets = torch.cat(all_act_targets, dim=0)

    # 5. Compute Metrics
    probs = torch.sigmoid(all_act_logits).numpy()
    targets = all_act_targets.numpy()
    preds = (probs > 0.5).astype(int)

    # Calculate global macro/micro metrics
    map_score = compute_mAP(all_act_logits, all_act_targets)
    
    # Flatten everything (micro metrics treating every label prediction across all videos independently)
    # or calculate macro metrics (averaging across all classes)
    
    # Micro Metrics
    micro_acc = accuracy_score(targets, preds)  # This is "exact match" subset accuracy for multi-label
    
    # A better accuracy for multi-label is flattened accuracy
    flat_targets = targets.flatten()
    flat_preds = preds.flatten()
    
    bin_acc = (flat_targets == flat_preds).mean() * 100
    micro_p = precision_score(targets, preds, average='micro', zero_division=0) * 100
    micro_r = recall_score(targets, preds, average='micro', zero_division=0) * 100
    macro_p = precision_score(targets, preds, average='macro', zero_division=0) * 100
    macro_r = recall_score(targets, preds, average='macro', zero_division=0) * 100

    print("\n--- TEST SET EVALUATION REPORT ---")
    print(f"Mean Average Precision (mAP) : {map_score:.2%}")
    print(f"Binary Classification Acc    : {bin_acc:.2f}% (probability > 0.5)")
    print(f"Micro Precision              : {micro_p:.2f}%")
    print(f"Micro Recall                 : {micro_r:.2f}%")
    print(f"Macro Precision              : {macro_p:.2f}%")
    print(f"Macro Recall                 : {macro_r:.2f}%")
    
    print("\nNote: 'Micro' aggregates contributions across all classes before calculating the metric.")
    print("'Macro' calculates the metric for each class independently, then averages them.")

if __name__ == "__main__":
    main()
