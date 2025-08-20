import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold
from sklearn.metrics import r2_score, mean_squared_error
from pymongo import MongoClient
from dotenv import load_dotenv

from src.models.sequence_autoencoder import SeqAutoencoder, cosine_loss

# =====================
# Dataset builder
# =====================
def build_dataset(k=1):
    client = MongoClient(os.getenv('MONGO_URL'))
    db = client["aestoria_app"]
    collection = db["training_images"]

    X, y, dump_ids = [], [], []
    sequences = {}

    for doc in collection.find().sort([("dump_id", 1), ("_id", 1)]):
        if "dump_id" not in doc:
            continue
        seq_id = doc["dump_id"]
        if seq_id not in sequences:
            sequences[seq_id] = []
        sequences[seq_id].append(doc["clip_embedding"])

    for dump_id, seq in sequences.items():
        if len(seq) <= k:
            continue
        for i in range(k, len(seq)):
            X.append(seq[i-k:i])
            y.append(seq[i])
            dump_ids.append(dump_id)

    return np.array(X), np.array(y), np.array(dump_ids)

# =====================
# Training loop
# =====================
def train_model(model, train_loader, device, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = cosine_loss(preds, yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model

# =====================
# Evaluation
# =====================
def evaluate_model(model, loader, name="Test"):
    model.eval()
    total_loss = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            total_loss += cosine_loss(preds, yb).item() * len(xb)
    avg_loss = total_loss / len(loader.dataset)
    print(f"{name} Loss: {avg_loss:.4f}")
    return avg_loss

# =====================
# 10-fold cross-validation
# =====================
def cross_validate(X, y, device, k_seq=1):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores, mse_scores, cos_losses = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1} ---")

        X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
        X_test, y_test = X[test_idx].to(device), y[test_idx].to(device)

        model = SeqAutoencoder(emb_dim=X.shape[2], k=k_seq).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train
        for epoch in range(10):  # shorter epochs for CV
            model.train()
            optimizer.zero_grad()
            preds = model(X_train)
            loss = cosine_loss(preds, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)

        cos_loss = cosine_loss(y_pred, y_test).item()
        y_true_np = y_test.cpu().numpy().reshape(len(test_idx), -1)
        y_pred_np = y_pred.cpu().numpy().reshape(len(test_idx), -1)

        r2 = r2_score(y_true_np, y_pred_np)
        mse = mean_squared_error(y_true_np, y_pred_np)

        print(f"Fold {fold+1} → Cosine: {cos_loss:.4f}, R²: {r2:.4f}, MSE: {mse:.4f}")
        cos_losses.append(cos_loss)
        r2_scores.append(r2)
        mse_scores.append(mse)

    print("\n===== Cross-validation Results =====")
    print(f"Mean Cosine: {np.mean(cos_losses):.4f} ± {np.std(cos_losses):.4f}")
    print(f"Mean R²:     {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Mean MSE:    {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")

# =====================
# Main
# =====================
if __name__ == "__main__":
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build dataset
    X, y, dump_ids = build_dataset(k=3)  # k-sequence
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    full_dataset = TensorDataset(X, y)

    # --- Image-level split
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset_img = Subset(full_dataset, train_idx)
    test_dataset_img  = Subset(full_dataset, test_idx)

    # --- Dump-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(np.arange(len(X)), groups=dump_ids))
    train_dataset_dump = Subset(full_dataset, train_idx)
    test_dataset_dump  = Subset(full_dataset, test_idx)

    # --- DataLoaders
    train_loader_img = DataLoader(train_dataset_img, batch_size=32, shuffle=True)
    test_loader_img  = DataLoader(test_dataset_img, batch_size=32, shuffle=False)
    train_loader_dump = DataLoader(train_dataset_dump, batch_size=32, shuffle=True)
    test_loader_dump  = DataLoader(test_dataset_dump, batch_size=32, shuffle=False)

    # --- Train
    model = SeqAutoencoder(emb_dim=512, k=3).to(device)
    model = train_model(model, train_loader_dump, device)  # use dump-level split for training

    # --- Evaluate
    evaluate_model(model, test_loader_img, "Image-level Test")
    evaluate_model(model, test_loader_dump, "Dump-level Test")

    # --- Cross-validation
    cross_validate(X, y, device, k_seq=3)

    # --- Save final model
    model_path = "src/models/autoencoder.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
