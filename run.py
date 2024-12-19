import torch
from torch.utils.data import random_split
from dataset import SymbolicDataset
from model.FAN import FAN
from model.MLP import MLP
from train import train_model

# Set Device
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define Models
fan_model = FAN(in_features=1, hidden_dim=32, num_layers=3).to(device)
mlp_model = MLP(in_features=1, hidden_dim=32, num_layers=3).to(device)


# Device check
print(f"Using device: {device}")
print(f"FAN model is on device: {next(fan_model.parameters()).device}")
print(f"MLP model is on device: {next(mlp_model.parameters()).device}")

# Datset
dataset = SymbolicDataset(num_samples = 1000, noise_level = 0.1)

# Splitting dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Train FAN Model
print('\n FAN model Trainining...')
fan_train_losses, fan_val_losses = train_model(train_dataset, val_dataset, device, fan_model, epochs=500, batch_size=32, lr=0.001, model_name="FAN")

print('\n MLP model Trainining...')
mlp_train_losses, mlp_val_losses = train_model(train_dataset, val_dataset, device, mlp_model, epochs=500, batch_size=32, lr=0.001, model_name="MLP")