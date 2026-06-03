
import os
import json
import glob
import random
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from torchvision import transforms

# -----------------------
# 1) GLOBAL CONFIGURATION
# -----------------------
DATA_ROOT     = ".\TMJ_sorted"
IMAGE_SIZE    = 224
IMAGE_H = IMAGE_SIZE
IMAGE_W = IMAGE_SIZE * 2
BATCH_SIZE    = 16
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 3
NUM_FOLDS     = 5
TEST_SIZE     = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------
# 2) ROI EXTRACTION
# -----------------------
def extract_condyle_rois(image_path, json_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with open(json_path, "r") as f:
        data = json.load(f)
    shapes = data.get("shapes", [])
    if len(shapes) < 2:
        raise ValueError(f"JSON {json_path} does not contain 2 polygons.")

    rois = []
    for shape in shapes:
        pts = shape["points"]
        arr = np.array(pts, dtype=np.int32)
        xs, ys = arr[:, 0], arr[:, 1]
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        centroid_x = xs.mean()
        rois.append({"bbox": (int(xmin), int(ymin), int(xmax), int(ymax)), "centroid_x": centroid_x})

    rois_sorted = sorted(rois, key=lambda x: x["centroid_x"])
    left_bbox, right_bbox = rois_sorted[0]["bbox"], rois_sorted[1]["bbox"]

    left_arr  = img_rgb[left_bbox[1]:left_bbox[3], left_bbox[0]:left_bbox[2]]
    right_arr = img_rgb[right_bbox[1]:right_bbox[3], right_bbox[0]:right_bbox[2]]

    return Image.fromarray(left_arr), Image.fromarray(right_arr)

# -----------------------
# 3) DATASET CONSTRUCTION
# -----------------------
def build_dataset(root_dir):
    pil_images = []
    y = []

    for folder in ["mild", "severe"]:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for ext in ("png", "jpg", "jpeg"):
            for img_path in glob.glob(os.path.join(folder_path, f"*.{ext}")):
                json_path = os.path.splitext(img_path)[0] + ".json"
                if not os.path.exists(json_path):
                    continue

                label = 0 if folder == "mild" else 1
                left_pil, right_pil = extract_condyle_rois(img_path, json_path)
                concat = Image.new('RGB', (left_pil.width + right_pil.width, left_pil.height))
                concat.paste(left_pil, (0, 0))
                concat.paste(right_pil, (left_pil.width, 0))
                pil_images.append(concat)
                y.append(label)

    return pil_images, np.array(y, dtype=np.int64)

# -----------------------
# 4) TRANSFORMATIONS
# -----------------------
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),      # radiographs are effectively gray
        transforms.Resize((IMAGE_H, IMAGE_W)),
        transforms.RandomHorizontalFlip(p=0.3),           # ok for bilateral-label classification
        transforms.RandomAffine(degrees=10, translate=(0.02,0.02), scale=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_H, IMAGE_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf

# -----------------------
# 5) CNN MODEL
# -----------------------
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_feats, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    )
    return model.to(DEVICE)

# -----------------------
# 6) EVALUATION
# -----------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss=correct=total=0
    preds_all, labels_all = [], []
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        logits1 = model(x)
        logits2 = model(torch.flip(x, dims=[3]))  # hflip on width
        logits  = 0.5*(logits1 + logits2)
        loss = criterion(logits, y)
        total_loss += loss.item()*x.size(0)
        preds = logits.argmax(1)
        correct += (preds==y).sum().item()
        total   += y.size(0)
        preds_all.extend(preds.cpu().numpy()); labels_all.extend(y.cpu().numpy())
    return total_loss/total, correct/total, np.array(preds_all), np.array(labels_all)

# -----------------------
# 7) CROSS-VALIDATION
# -----------------------
def cross_validate(pil_images, y):
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    transform_train, transform_val = get_transforms()
    acc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(pil_images, y), 1):
        print(f"\n--- Fold {fold}/{NUM_FOLDS} ---")
        X_train = torch.stack([transform_train(pil_images[i]) for i in train_idx])
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        X_val   = torch.stack([transform_val(pil_images[i]) for i in val_idx])
        y_val   = torch.tensor(y[val_idx], dtype=torch.long)

        weights = torch.tensor([1.0 / (y_train == i).sum().item() for i in y_train], dtype=torch.float)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
        val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model = get_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_acc, patience_counter = 0, 0
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
            print(f"Epoch {epoch} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
            scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping.")
                    break

        model.load_state_dict(best_state)
        _, _, final_preds, final_trues = evaluate(model, val_loader, criterion)
        print("Confusion matrix:")
        print(confusion_matrix(final_trues, final_preds))
        acc_scores.append(best_acc)

    print("\n=== Cross-Validation Results ===")
    for i, score in enumerate(acc_scores, 1):
        print(f"Fold {i}: {score:.4f}")
    print(f"Average: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")

# -----------------------
# 8) FINAL TEST EVALUATION
# -----------------------
def final_test(X_test, y_test):
    print("\n=== Final Test Set Evaluation ===")
    transform_train, transform_val = get_transforms()
    X_test_tensor = torch.stack([transform_val(img) for img in X_test])
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE)

    model = get_model()
    X_train_tensor = torch.stack([transform_train(img) for img in X_trainval])
    y_train_tensor = torch.tensor(y_trainval, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc, patience_counter = 0, 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch} | Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    _, _, preds, trues = evaluate(model, test_loader, criterion)
    print("\nConfusion matrix - Final test:")
    print(confusion_matrix(trues, preds))
    print(f"Final Test Accuracy: {best_acc:.4f}")

# -----------------------
# 9) MAIN FUNCTION
# -----------------------
def main():
    pil_images, y = build_dataset(DATA_ROOT)
    print(f"Total images: {len(pil_images)}")
    print(f"Healthy: {(y == 0).sum()} | Pathological: {(y == 1).sum()}")

    global X_trainval, X_test, y_trainval, y_test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        pil_images, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    print(f"Training + Validation set: {len(X_trainval)} images")
    print(f"Test set: {len(X_test)} images")

    num_trainval_healthy = (y_trainval == 0).sum()
    num_trainval_patho = (y_trainval == 1).sum()
    num_test_healthy = (y_test == 0).sum()
    num_test_patho = (y_test == 1).sum()

    print("\n--- Class Distribution ---")
    print(f"Train+Val set: {num_trainval_healthy} healthy | {num_trainval_patho} pathological")
    print(f"    → Ratio: {100 * num_trainval_healthy / len(y_trainval):.1f}% healthy, {100 * num_trainval_patho / len(y_trainval):.1f}% pathological")
    print(f"Test set    : {num_test_healthy} healthy | {num_test_patho} pathological")
    print(f"    → Ratio: {100 * num_test_healthy / len(y_test):.1f}% healthy, {100 * num_test_patho / len(y_test):.1f}% pathological")
    cross_validate(X_trainval, y_trainval)
    final_test(X_test, y_test)

if __name__ == "__main__":
    main()
