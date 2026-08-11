"""training_pipeline.py - High-level training functions"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os
import pandas as pd

# You'll need to move these to appropriate files:
from model import AttentionLSTM
from custom_datasets import SepsisDataset


def evaluate_model(model, test_loader, device):
    """
    Evaluate a trained model on test data
    
    Returns:
        dict: Dictionary containing predictions, labels, and all metrics
    """
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs, _ = model(batch_X)
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(batch_y.numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    # Calculate metrics
    auc = roc_auc_score(test_labels, test_preds)
    binary_preds = (test_preds > 0.5).astype(int)
    
    metrics = {
        'predictions': test_preds,
        'labels': test_labels,
        'auc': auc,
        'accuracy': accuracy_score(test_labels, binary_preds),
        'precision': precision_score(test_labels, binary_preds, zero_division=0),
        'recall': recall_score(test_labels, binary_preds),
        'f1': f1_score(test_labels, binary_preds)
    }
    
    return metrics


def train_and_evaluate_single_model(sequences, hours_ahead, model_class, 
                                  epochs=50, batch_size=32, save_dir='models'):
    """
    Train and evaluate a single model for a specific prediction horizon
    
    Args:
        sequences: Dictionary containing train/test sequences
        hours_ahead: Prediction horizon (2, 4, or 6)
        model_class: Model class to instantiate (e.g., AttentionLSTM)
        epochs: Number of training epochs
        batch_size: Batch size for training
        save_dir: Directory to save models
        
    Returns:
        dict: Results including model, history, and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training for {hours_ahead}-hour early detection")
    print(f"{'='*60}")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    X_train = sequences[hours_ahead]['X_train']
    y_train = sequences[hours_ahead]['y_train']
    X_test = sequences[hours_ahead]['X_test']
    y_test = sequences[hours_ahead]['y_test']
    
    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create data loaders
    train_dataset = SepsisDataset(X_train_split, y_train_split)
    val_dataset = SepsisDataset(X_val_split, y_val_split)
    test_dataset = SepsisDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[2]
    print(f"Input size: {input_size}")
    model = model_class(input_size)
    
    # Train model
    trained_model, history = AttentionLSTM.train_model(model, train_loader, val_loader, epochs=epochs)
    
    # Evaluate model
    metrics = evaluate_model(trained_model, test_loader, device)
    
    # Prepare results
    results = {
        'model': trained_model,
        'history': history,
        **metrics  # Unpack all metrics
    }
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'model_{hours_ahead}h.pth')
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Print results
    print(f"\nTest Results ({hours_ahead}h early detection):")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    
    return results


def train_all_models(sequences, model_class, prediction_hours=[2, 4, 6], **kwargs):
    """
    Train models for all prediction horizons
    
    Args:
        sequences: Dictionary containing sequences for each prediction horizon
        model_class: Model class to use
        prediction_hours: List of prediction horizons
        **kwargs: Additional arguments passed to train_and_evaluate_single_model
        
    Returns:
        dict: Results for all prediction horizons
    """
    results = {}
    
    for hours in prediction_hours:
        results[hours] = train_and_evaluate_single_model(
            sequences, hours, model_class, **kwargs
        )
    
    return results

# Wrapper function to load data for ablation training
def train_auc_wrapper(X, y, input_size, device='cpu'):
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    train_loader = DataLoader(SepsisDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(SepsisDataset(X_val, y_val), batch_size=64)

    model = AttentionLSTM(input_size=input_size)
    model, history = AttentionLSTM.train_model(model, train_loader, val_loader, epochs=10, lr=1e-3)

    # In train_auc_wrapper, before returning:
    auc_value = history['val_auc'][-1]
    return model, auc_value

def run_feature_ablation(df, feature_cols, create_fn, train_fn, 
                        lookback_hours=6, prediction_ahead_hours=2, device='cpu'):
    """Run leave-one-feature-out ablation
    Args:
        df: DataFrame with patient data
        feature_cols: List of feature names to ablate
        create_fn: Function to create sequences (e.g., create_sequences_for_classification)
        train_fn: Function to train the model (e.g., train_attention_model)
        lookback_hours: Hours of historical data to use
        prediction_ahead_hours: How many hours before the end to make predictions
        device: Device to run the model on ('cpu' or 'cuda')
        Returns:
            DataFrame with feature importance results"""
    # 1. Full-feature baseline
    X_full, y_full, *_ = create_fn(df, lookback_hours, prediction_ahead_hours)
    input_size = X_full.shape[2]
    
    # Train and save baseline
    base_model, base_auc = train_fn(X_full, y_full, input_size=input_size, device=device)

    #Debugging: Print baseline AUC
    #print(f"Base AUC type: {type(base_auc)}, value: {base_auc}")
    # Save baseline model
    #torch.save(base_model.state_dict(), f'ablation_{prediction_ahead_hours}h_baseline.pth')
    
    # 2. Leave-one-feature-out loop
    results = []
    for feat in feature_cols:
        reduced_df = df.drop(columns=feat)
        try:
            X_reduced, y_reduced, *_ = create_fn(reduced_df, lookback_hours, prediction_ahead_hours)
            if X_reduced.shape[0] == 0:
                continue
            
            # Train and save model without this feature
            model, auc = train_fn(X_reduced, y_reduced, input_size=X_reduced.shape[2], device=device)
            # Save ablation model
            #   torch.save(model.state_dict(), f'ablation_{prediction_ahead_hours}h_without_{feat}.pth')
            
            results.append({
                'Feature': feat,
                'AUC_without': auc,
                'Delta_AUC': base_auc - auc
            })
        except Exception as e:
            print(f"Skipping {feat}: {e}")
            continue
    
    #models_dict = {'baseline': base_model}  # If you want to return models
    return pd.DataFrame(results).sort_values('Delta_AUC', ascending=False)#, models_dict

# Summary printing function
def print_results_summary(results):
    """Print a summary table of all results"""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Hours':<10} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-"*80)
    
    for hours in sorted(results.keys()):
        r = results[hours]
        print(f"{hours:<10} {r['auc']:<10.4f} {r['accuracy']:<10.4f} "
              f"{r['precision']:<10.4f} {r['recall']:<10.4f} {r['f1']:<10.4f}")