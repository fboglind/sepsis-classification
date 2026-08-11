"""lstm_model.py"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class AttentionLSTM(nn.Module):
    """LSTM with attention for sepsis classification
    Args:
        input_size (int): Size of the input features
        hidden_size (int): Size of the hidden layer in LSTM
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate for regularization
    Returns:
        output (torch.Tensor): Output tensor of shape (batch_size, 1) with sigmoid activation
        attention_weights (torch.Tensor): Attention weights of shape (batch_size, sequence_length)
    """


    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(AttentionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(attended)
        return output.squeeze(), attention_weights.squeeze()

    @staticmethod
    def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
        """Train the model
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            lr: Learning rate for the optimizer
        Returns:
            model: Trained PyTorch model
            history: Dictionary containing training and validation loss and AUC scores"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
        best_val_auc = 0
        best_model_state = None
        patience = 0
        max_patience = 10
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            train_preds = []
            train_labels = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())
            
            # Validation
            model.eval()
            val_losses = []
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs, _ = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_losses.append(loss.item())
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_auc = roc_auc_score(train_labels, train_preds)
            val_auc = roc_auc_score(val_labels, val_preds)
            
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(np.mean(val_losses))
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)
            
            # Learning rate scheduling
            scheduler.step(val_auc)
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        return model, history

# Define a custom PyTorch Dataset for Sepsis time series data
class SepsisDataset(Dataset):
    """Dataset for sepsis classification"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]