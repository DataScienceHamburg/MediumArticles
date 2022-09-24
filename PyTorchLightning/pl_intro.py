#%% packages
# data handling
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#%% data import
data = fetch_california_housing()
X = data.data.astype(np.float32)
y = data.target
print(f"X shape: {X.shape}, y shape: {y.shape}")
scaler = StandardScaler()  # data scaling
X_scaled = scaler.fit_transform(X)
#%% splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=123, train_size=0.8)
#%% Dataset and Dataloader
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
#%%
class LitLinearRegression(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden1, learning_rate):
        super(LitLinearRegression, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden1, hidden1)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden1, output_dim)
        self.relu = nn.ReLU()
        self.loss_fun = nn.MSELoss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y = y.type(torch.float32)
        # forward pass
        y_pred = self.forward(X).squeeze()
        # compute loss
        loss = self.loss_fun(y_pred, y)
        self.log_dict({'train_loss': loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        X, y = test_batch
        y = y.type(torch.float32)
        # forward pass
        y_pred = self.forward(X).squeeze()        
        # compute metrics       
        print(y_pred) 
        r2 = r2_score(y_pred, y)
        loss = self.loss_fun(y_pred[0], y[0])
        self.log_dict({'test_loss': loss, 'r2': r2}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

#%% model instance and training
#------------------------------
#%% Hyperparameters
hidden1 = 20
max_epochs = 500
lr = 0.1
train_batch_size = 128
test_batch_size =8192
# model instance
input_dim = X_scaled.shape[1]
model = LitLinearRegression(input_dim=input_dim, output_dim=1, hidden1=hidden1, learning_rate=lr)
#%% Callbacks
early_stop_callback = EarlyStopping(monitor="train_loss", min_delta=0.00, patience=20, verbose=True, mode="min")
# training
train_loader = DataLoader(dataset = LinearRegressionDataset(X_train, y_train), batch_size=train_batch_size)
test_loader = DataLoader(dataset = LinearRegressionDataset(X_test, y_test), batch_size=test_batch_size)
trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=max_epochs, callbacks=[early_stop_callback], log_every_n_steps=8)
trainer.fit(model=model, train_dataloaders=train_loader)
# %% testing
trainer.test(model=model, dataloaders=test_loader)
