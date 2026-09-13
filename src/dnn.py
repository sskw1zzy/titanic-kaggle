import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim


class DNNModel(nn.Module):
    """Configurable MLP: any number of hidden layers, optional BatchNorm/Dropout,
    switchable activation. Outputs a single raw logit (binary classification)."""
    
    def __init__(self, input_size, hidden_sizes, dropout_p=0.0, use_batchnorm=False, activation='relu'):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(self._get_activation(activation))
            self.layers.append(nn.Dropout(dropout_p))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, 1)            
        
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'elu':
            return nn.ELU()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

def prepare_dnn_data(X, y, batch_size):
    """Wraps X/y into a shuffled DataLoader of float32 tensors."""
    X_tensor = torch.tensor(X.astype('float32').values, dtype=torch.float32)
    y_tensor = torch.tensor(y.astype('float32').values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_dnn(model, loader, loss_fn, optimizer, num_epochs, scheduler=None):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if scheduler is not None:
            scheduler.step()
     
    return model

def evaluate_dnn(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
            predicted_labels = (torch.sigmoid(predictions) > 0.5).float()
            correct += (predicted_labels == y_batch).sum().item()
            total += y_batch.size(0)
            
    return correct / total

def evaluate_dnn_kfold(X, y, seed, n_splits, model_params, train_params, optimizer_name='adam', scheduler_name=None):
    """Stratified K-Fold CV accuracy for DNNModel with configurable optimizer/scheduler."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_loader = prepare_dnn_data(X_train, y_train, batch_size=train_params['batch_size'])
        val_loader = prepare_dnn_data(X_val, y_val, batch_size=train_params['batch_size'])
        
        model = DNNModel(input_size=X.shape[1], **model_params)
        loss_fn = nn.BCEWithLogitsLoss()
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=train_params['lr'])
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=train_params['lr'])
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=train_params['lr'], momentum=0.9)
        
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['num_epochs'])
        else:
            scheduler = None
        
        model = train_dnn(model, train_loader, loss_fn, optimizer, train_params['num_epochs'], scheduler=scheduler)   
        val_accuracy = evaluate_dnn(model, val_loader)
        scores.append(val_accuracy)
        
    return scores