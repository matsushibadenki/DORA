# ファイルパス: benchmarks/stability_benchmark.py
import sys
import os
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import argparse
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from snn_research.models.visual_cortex import VisualCortex
from snn_research.utils.observer import NeuromorphicObserver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StabilityBenchmark")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def flatten_tensor(x):
    return torch.flatten(x)

def get_robust_dataset(root_dir, train=True, transform=None):
    try:
        dataset = datasets.MNIST(root_dir, train=train, download=True, transform=transform)
        return dataset
    except Exception as e:
        logger.warning(f"Using fallback dataset: {e}")
        digits = load_digits()
        X = digits.data
        y = digits.target
        split_idx = int(len(X) * 0.8)
        if train:
            X_data, y_data = X[:split_idx], y[:split_idx]
        else:
            X_data, y_data = X[split_idx:], y[split_idx:]
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.long)
        X_padded = torch.zeros(X_tensor.shape[0], 784)
        X_padded[:, :64] = X_tensor
        X_padded = X_padded / 16.0
        return TensorDataset(X_padded, y_tensor)

def generate_negative_data(data, device):
    batch_size, dim = data.shape
    perm = torch.randperm(dim).to(device)
    noise = data[:, perm]
    return noise

def get_model_features(model, loader, device):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            out = model(data, phase="wake", update_weights=False)
            if 'spikes' in out and len(out['spikes']) > 0:
                layer_activities = []
                for name in sorted(out['spikes'].keys()):
                    if name == "Retina": continue
                    act = out['spikes'][name].float().cpu().numpy()
                    layer_activities.append(act)
                if layer_activities:
                    batch_features = np.concatenate(layer_activities, axis=1)
                    features.append(batch_features)
                    labels.append(target.numpy())
    if len(features) == 0: return np.array([]), np.array([])
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def run_benchmark(config):
    observer = NeuromorphicObserver(experiment_name="stability_benchmark")
    observer.set_config(vars(config))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"
    logger.info(f"🚀 Starting Stability Benchmark on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])
    
    data_dir = os.path.join(project_root, 'data')
    full_train_dataset = get_robust_dataset(data_dir, train=True, transform=transform)
    full_test_dataset = get_robust_dataset(data_dir, train=False, transform=transform)

    # 10000 samples for better signal
    subset_size = min(10000, len(full_train_dataset))
    train_subset = Subset(full_train_dataset, list(range(subset_size)))
    test_subset = Subset(full_test_dataset, list(range(min(1000, len(full_test_dataset)))))

    # Batch Size 32 (Safe for 3000 dim with detach)
    batch_size = 32
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    feature_train_loader = DataLoader(train_subset, batch_size=64, shuffle=False)
    feature_test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    stability_scores = []

    for run in range(config.num_runs):
        seed = 42 + run
        set_seed(seed)
        
        if 'model' in locals(): del model
        gc.collect()
        if device == "mps": torch.mps.empty_cache()

        model = VisualCortex(device=torch.device(device))
        model.to(device)

        logger.info(f"Run {run+1}/{config.num_runs}: Training Start (Data Size: {subset_size})")

        for epoch in range(config.epochs):
            model.train()
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                
                if batch_idx % 50 == 0:
                    if device == "mps": torch.mps.empty_cache()

                _ = model(data, phase="wake", update_weights=True)
                
                noise = generate_negative_data(data, device)
                _ = model(noise, phase="sleep", update_weights=True)

            if (epoch+1) % 2 == 0 or epoch == config.epochs - 1:
                stats = model.get_goodness()
                # Print explicit Goodness
                v1_g = stats.get("V1_goodness", 0.0)
                print(f"Run {run+1} Epoch {epoch+1} STATS: V1_Goodness={v1_g:.4f}")
                logger.info(f"Run {run+1} Epoch {epoch+1}: Goodness={v1_g:.4f}")

        logger.info("Evaluating...")
        X_train, y_train = get_model_features(model, feature_train_loader, device)
        X_test, y_test = get_model_features(model, feature_test_loader, device)
        
        if len(X_train) > 0:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(
                max_iter=5000, 
                solver='lbfgs', 
                multi_class='multinomial',
                C=1.0, 
                random_state=seed,
                n_jobs=-1
            )
            clf.fit(X_train_scaled, y_train)
            
            test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
            stability_scores.append(test_acc)
            
            logger.info(f"Run {run+1} Result: Stability Score = {test_acc*100:.2f}%")
        else:
            stability_scores.append(0.0)

    mean_stability = np.mean(stability_scores) * 100
    summary = f"Benchmark Finished. Mean Stability: {mean_stability:.2f}%"
    print("\n" + "="*50)
    print(summary)
    print("="*50)
    observer.save_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=1)
    # [RECOMMENDATION] 10 epochs for convergence
    parser.add_argument("--epochs", type=int, default=10) 
    args = parser.parse_args()
    run_benchmark(args)