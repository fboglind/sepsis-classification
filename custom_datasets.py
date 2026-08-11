"""datasets.py - Dataset classes and data preparation functions"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SepsisDataset(Dataset):
    """PyTorch Dataset for sepsis time series data"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    
def prepare_all_sequences(train_data, test_data, prediction_hours=[2, 4, 6], lookback_hours=6):
    """
    Prepare sequences for all prediction horizons
    
    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        prediction_hours: List of prediction horizons
        lookback_hours: Hours of historical data to use
        
    Returns:
        dict: Dictionary with sequences for each prediction horizon
    """
    sequences = {}
    
    for hours_ahead in prediction_hours:
        print(f"\nCreating sequences for {hours_ahead}-hour early detection...")
        
        X_train, y_train, train_ids, train_times = create_sequences_for_classification(
            train_data,
            lookback_hours=lookback_hours,
            prediction_ahead_hours=hours_ahead
        )
        
        X_test, y_test, test_ids, test_times = create_sequences_for_classification(
            test_data,
            lookback_hours=lookback_hours,
            prediction_ahead_hours=hours_ahead
        )
        
        sequences[hours_ahead] = {
            'X_train': X_train, 'y_train': y_train,
            'train_ids': train_ids, 'train_times': train_times,
            'X_test': X_test, 'y_test': y_test,
            'test_ids': test_ids, 'test_times': test_times
        }
        
        print(f"  Training sequences: {len(X_train)} from {len(np.unique(train_ids))} patients")
        print(f"  Test sequences: {len(X_test)} from {len(np.unique(test_ids))} patients")
        print(f"  Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
    
    return sequences


def create_sequences_for_classification(data, lookback_hours=6, prediction_ahead_hours=2):
    """
    Create sequences for sepsis classification
    
    Args:
        data: DataFrame with patient data
        lookback_hours: Hours of historical data to use
        prediction_ahead_hours: How many hours before the end to make predictions
    
    Returns:
        X: Input sequences
        y: Binary labels (patient has sepsis or not)
        patient_ids: Corresponding patient IDs
        time_points: Time points where predictions are made
    """
    # Convert hours to timesteps
    lookback_steps = lookback_hours * 2
    ahead_steps = prediction_ahead_hours * 2
    
    # Features to use
    feature_cols = [col for col in data.columns 
                   if col not in ['id', 'sepsis', 'severity', 'timestep']]
    
    X, y, patient_ids, time_points = [], [], [], []
    
    # Process each patient
    for patient_id, patient_data in data.groupby('id'):
        patient_data = patient_data.sort_values('timestep').reset_index(drop=True)
        
        # Get the patient's sepsis label
        patient_sepsis = patient_data['sepsis'].iloc[0]
        
        # Skip if not enough data
        if len(patient_data) < lookback_steps + ahead_steps:
            continue
        
        # Create sequences at different time points
        max_start = len(patient_data) - lookback_steps - ahead_steps
        
        # Sample multiple sequences from this patient's timeline
        n_samples = min(5, max_start)
        
        if n_samples > 0:
            sample_points = np.linspace(0, max_start-1, n_samples, dtype=int)
            
            for start_idx in sample_points:
                # Extract sequence
                sequence = patient_data.iloc[start_idx:start_idx+lookback_steps][feature_cols].values
                
                X.append(sequence)
                y.append(patient_sepsis)
                patient_ids.append(patient_id)
                time_points.append(start_idx + lookback_steps)
    
    return np.array(X), np.array(y), np.array(patient_ids), np.array(time_points)