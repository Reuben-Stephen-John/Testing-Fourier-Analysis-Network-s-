import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(train_dataset, val_dataset, device, model, epochs, batch_size, lr, model_name):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # MSE Loss
    criterion = nn.MSELoss()

    # Adam Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    # Use tqdm with dynamic descriptions
    progress_bar = tqdm(range(1, epochs+1), desc=f'Training {model_name}', leave=True)

    for epoch in progress_bar:
        # Train
        model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward Pass
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x_batch.size(0)

        # Validation 
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                epoch_val_loss += loss.item() * x_batch.size(0)

        # Avg Loss
        epoch_train_loss /= len(train_dataset)
        epoch_val_loss /= len(val_dataset)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Update tqdm description and print every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            progress_bar.set_postfix({
                "Epoch": f"{epoch}/{epochs}",
                "Train Loss": f"{epoch_train_loss:.6f}",
                "Val Loss": f"{epoch_val_loss:.6f}"
            })

    return train_losses, val_losses
