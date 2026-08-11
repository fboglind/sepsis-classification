"""Utility functions"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def save_all_results(results, ablation_results, save_dir='saved_results'):
    """Save all results to avoid retraining
    Args:
        results: Dictionary with main results (e.g., AUC, accuracy)
        ablation_results: DataFrame with feature importance results
        save_dir: Directory to save results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save main results (without the model objects to keep file size small)
    main_results_to_save = {}
    for hours in results.keys():
        main_results_to_save[hours] = {
            'auc': results[hours]['auc'],
            'accuracy': results[hours]['accuracy'],
            'precision': results[hours]['precision'],
            'recall': results[hours]['recall'],
            'f1': results[hours]['f1'],
            'predictions': results[hours]['predictions'],
            'labels': results[hours]['labels']
        }
    
    with open(os.path.join(save_dir, 'main_results.pkl'), 'wb') as f:
        pickle.dump(main_results_to_save, f)
    
    # Save ablation results
    with open(os.path.join(save_dir, 'ablation_results.pkl'), 'wb') as f:
        pickle.dump(ablation_results, f)
    
    print(f"Results saved to {save_dir}/")

def load_all_results(save_dir='saved_results'):
    """Load saved results for plotting
    Args:
        save_dir: Directory where results are saved
    Returns:
        main_results: Dictionary with main results"""
    with open(os.path.join(save_dir, 'main_results.pkl'), 'rb') as f:
        main_results = pickle.load(f)
    
    with open(os.path.join(save_dir, 'ablation_results.pkl'), 'rb') as f:
        ablation_results = pickle.load(f)
    
    print("Results loaded successfully!")
    return main_results, ablation_results

def plot_ablation_results(ablation_result,h):
    """Plot feature importance from ablation results
    Args:
        ablation_result: DataFrame with feature importance results
        h: Prediction horizon (e.g., 2, 4, 6 hours)
    Returns:
        None: Displays a horizontal bar plot of feature importance
    """
    top_features = ablation_result.sort_values("Delta_AUC", ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Delta_AUC'])
    plt.xlabel("Δ AUC when feature is removed")
    plt.title(f"Top Feature Importance via Ablation for {h}")
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    plt.show()

def analyze_errors(results, threshold=0.5):
    """
    Analyze false positives and false negatives to understand model mistakes. Helps identify:
    - Is the model too conservative (many false negatives)?
    - Is the model too aggressive (many false positives)?
    - Should we adjust the decision threshold?
    """
    print("ERROR ANALYSIS")
    print("="*60)
    
    for hours in [2, 4, 6]:
        preds = results[hours]['predictions']
        labels = results[hours]['labels']
        binary_preds = (preds > threshold).astype(int)
        
        # Find errors
        fp_mask = (binary_preds == 1) & (labels == 0)  # Predicted sepsis, actually no sepsis
        fn_mask = (binary_preds == 0) & (labels == 1)  # Predicted no sepsis, actually sepsis
        tp_mask = (binary_preds == 1) & (labels == 1)  # Correctly predicted sepsis
        tn_mask = (binary_preds == 0) & (labels == 0)  # Correctly predicted no sepsis
        
        print(f"\n{hours}-hour prediction (threshold={threshold}):")
        print(f"  True Positives:  {tp_mask.sum()} ({tp_mask.sum()/len(labels)*100:.1f}%)")
        print(f"  True Negatives:  {tn_mask.sum()} ({tn_mask.sum()/len(labels)*100:.1f}%)")
        print(f"  False Positives: {fp_mask.sum()} ({fp_mask.sum()/len(labels)*100:.1f}%) - False alarms")
        print(f"  False Negatives: {fn_mask.sum()} ({fn_mask.sum()/len(labels)*100:.1f}%) - Missed cases")
        
        # Clinical insight
        if fn_mask.sum() > fp_mask.sum():
            print("  → Model is too conservative (missing too many sepsis cases)")
        elif fp_mask.sum() > fn_mask.sum() * 2:
            print("  → Model is too aggressive (too many false alarms)")
        else:
            print("  → Model has reasonable balance")

def visualize_attention_weights(model, X_sample, feature_names, device ,patient_idx=0):
    """
    Visualize what time points the model pays attention to
    
    This helps understand:
    - Does the model focus on recent data or look at the full history?
    - Are there specific time patterns it considers important?
    """
    model.eval()
    
    # Get a single patient sequence
    X_patient = torch.FloatTensor(X_sample[patient_idx:patient_idx+1]).to(device)
    
    with torch.no_grad():
        _, attention_weights = model(X_patient)
    
    if attention_weights is None:
        print("This model doesn't have attention weights")
        return
    
    # Convert to numpy
    weights = attention_weights.squeeze().cpu().numpy()
    
    # Plot attention over time
    plt.figure(figsize=(10, 4))
    timesteps = np.arange(len(weights))
    plt.plot(timesteps, weights, 'b-', linewidth=2)
    plt.fill_between(timesteps, weights, alpha=0.3)
    plt.xlabel('Time Steps (30-min intervals)')
    plt.ylabel('Attention Weight')
    plt.title(f'Attention Weights Over Time - Patient {patient_idx}')
    plt.grid(True, alpha=0.3)
    
    # Add interpretation
    max_attention_time = np.argmax(weights)
    plt.axvline(max_attention_time, color='red', linestyle='--', alpha=0.5)
    plt.text(max_attention_time+0.5, np.max(weights)*0.9, 
             f'Peak attention\nat t={max_attention_time}', color='red')
    
    plt.tight_layout()
    plt.show()
    
    # Show which features at the peak attention time
    print(f"\nAt peak attention time (t={max_attention_time}):")
    feature_values = X_sample[patient_idx, max_attention_time, :]
    top_indices = np.argsort(np.abs(feature_values))[-5:]  # Top 5 features by absolute value
    
    print("Top features by value:")
    for idx in reversed(top_indices):
        print(f"  {feature_names[idx]}: {feature_values[idx]:.3f}")

def analyze_thresholds(results):
    """
    Test different probability thresholds to find the best balance
    
    Important for clinical use: 
    - Lower threshold = catch more sepsis cases but more false alarms
    - Higher threshold = fewer false alarms but might miss cases
    """   
    thresholds = np.arange(0.1, 0.9, 0.05)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, hours in enumerate([2, 4, 6]):
        preds = results[hours]['predictions']
        labels = results[hours]['labels']
        
        f1_scores = []
        recalls = []
        precisions = []
        
        for thresh in thresholds:
            binary_preds = (preds > thresh).astype(int)
            f1_scores.append(f1_score(labels, binary_preds))
            recalls.append(recall_score(labels, binary_preds))
            precisions.append(precision_score(labels, binary_preds, zero_division=0))
        
        ax = axes[idx]
        ax.plot(thresholds, f1_scores, 'b-', label='F1 Score')
        ax.plot(thresholds, recalls, 'g-', label='Recall (Sensitivity)')
        ax.plot(thresholds, precisions, 'r-', label='Precision')
        
        # Mark optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        ax.axvline(thresholds[optimal_idx], color='black', linestyle='--', alpha=0.5)
        ax.text(thresholds[optimal_idx]+0.02, 0.5, 
                f'Optimal\n{thresholds[optimal_idx]:.2f}', fontsize=10)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'{hours}-hour Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


def analyze_class_distribution(y_train, y_test):
    """
    Visualize class distribution in training and test sets
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training set
    train_counts = np.bincount(y_train.astype(int))
    train_percentages = train_counts / len(y_train) * 100
    
    bars1 = ax1.bar(['No Sepsis', 'Sepsis'], train_counts, color=['skyblue', 'salmon'])
    ax1.set_title('Training Set Distribution', fontsize=14)
    ax1.set_ylabel('Number of Sequences')
    
    # Add value labels on bars
    for bar, count, pct in zip(bars1, train_counts, train_percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
    
    # Test set
    test_counts = np.bincount(y_test.astype(int))
    test_percentages = test_counts / len(y_test) * 100
    
    bars2 = ax2.bar(['No Sepsis', 'Sepsis'], test_counts, color=['skyblue', 'salmon'])
    ax2.set_title('Test Set Distribution', fontsize=14)
    ax2.set_ylabel('Number of Sequences')
    
    for bar, count, pct in zip(bars2, test_counts, test_percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print imbalance ratios
    print("Class Imbalance Analysis:")
    print(f"Training: {train_counts[0]/train_counts[1]:.2f}:1 (No Sepsis:Sepsis)")
    print(f"Test: {test_counts[0]/test_counts[1]:.2f}:1 (No Sepsis:Sepsis)")


def plot_feature_distributions(data, feature_cols, n_features=6):
    """
    Plot distributions of top features for sepsis vs non-sepsis cases
    """
    # Select top features to plot (you might want to select based on importance)
    features_to_plot = feature_cols[:n_features]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        
        # Separate data by sepsis status
        sepsis_values = data[data['sepsis'] == 1][feature].dropna()
        no_sepsis_values = data[data['sepsis'] == 0][feature].dropna()
        
        # Plot distributions
        ax.hist(no_sepsis_values, bins=30, alpha=0.6, label='No Sepsis', 
                density=True, color='skyblue', edgecolor='black')
        ax.hist(sepsis_values, bins=30, alpha=0.6, label='Sepsis', 
                density=True, color='salmon', edgecolor='black')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add median lines
        median_no_sepsis = no_sepsis_values.median()
        median_sepsis = sepsis_values.median()
        ax.axvline(median_no_sepsis, color='blue', linestyle='--', alpha=0.7)
        ax.axvline(median_sepsis, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical comparison
    print("\nStatistical Summary (Median values):")
    print("-" * 60)
    for feature in features_to_plot:
        sepsis_median = data[data['sepsis'] == 1][feature].median()
        no_sepsis_median = data[data['sepsis'] == 0][feature].median()
        print(f"{feature:20} | No Sepsis: {no_sepsis_median:8.2f} | Sepsis: {sepsis_median:8.2f}")


def plot_model_comparison(results):
    """
    Compare model performance across different prediction horizons
    """
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    hours_list = sorted(results.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Extract values for each prediction horizon
        values = [results[h][metric] for h in hours_list]
        
        # Create bar plot
        bars = ax.bar([f'{h}h' for h in hours_list], values, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Prediction Horizon')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    # Add overall title
    fig.suptitle('Model Performance Comparison Across Prediction Horizons', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Also create a single comparison plot
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(hours_list))
    width = 0.15
    
    for i, metric in enumerate(['auc', 'recall', 'precision', 'f1']):
        values = [results[h][metric] for h in hours_list]
        plt.bar(x + i*width, values, width, label=metric.upper())
    
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Score')
    plt.title('Key Metrics Comparison')
    plt.xticks(x + width*1.5, [f'{h} hours' for h in hours_list])
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3, axis='y')
    
    for i, h in enumerate(hours_list):
        for j, metric in enumerate(['auc', 'recall', 'precision', 'f1']):
            val = results[h][metric]
            plt.text(i + j*width, val + 0.01, f'{val:.2f}', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_temporal_patterns(data, feature_cols, n_features=4):
    """
    Visualize how features change over time for sepsis vs non-sepsis patients
    """
    # Sample a few patients from each group
    sepsis_patients = data[data['sepsis'] == 1]['id'].unique()[:3]
    no_sepsis_patients = data[data['sepsis'] == 0]['id'].unique()[:3]
    
    features_to_plot = feature_cols[:n_features]
    
    fig, axes = plt.subplots(n_features, 2, figsize=(15, 3*n_features))
    
    for feat_idx, feature in enumerate(features_to_plot):
        # Plot sepsis patients
        ax_sepsis = axes[feat_idx, 0]
        for patient_id in sepsis_patients:
            patient_data = data[data['id'] == patient_id].sort_values('timestep')
            ax_sepsis.plot(patient_data['timestep'], patient_data[feature], 
                          alpha=0.7, label=f'Patient {patient_id}')
        ax_sepsis.set_title(f'{feature} - Sepsis Patients')
        ax_sepsis.set_xlabel('Time Step')
        ax_sepsis.set_ylabel(feature)
        ax_sepsis.grid(True, alpha=0.3)
        
        # Plot non-sepsis patients
        ax_no_sepsis = axes[feat_idx, 1]
        for patient_id in no_sepsis_patients:
            patient_data = data[data['id'] == patient_id].sort_values('timestep')
            ax_no_sepsis.plot(patient_data['timestep'], patient_data[feature], 
                             alpha=0.7, label=f'Patient {patient_id}')
        ax_no_sepsis.set_title(f'{feature} - Non-Sepsis Patients')
        ax_no_sepsis.set_xlabel('Time Step')
        ax_no_sepsis.set_ylabel(feature)
        ax_no_sepsis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

