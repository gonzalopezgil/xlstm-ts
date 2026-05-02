# src/ml/models/xlstm_ts/xlstm_ts_model.py

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

# -------------------------------------------------------------------------------------------
# Data loader
# -------------------------------------------------------------------------------------------

# DataLoader for batching
def create_dataloader(x, y, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# -------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------

def train_model(xlstm_stack, input_projection, output_projection, train_x, train_y, val_x, val_y):
    # Hyperparameters
    learning_rate = 0.0001
    num_epochs = int(os.environ.get("XLSTM_TS_NUM_EPOCHS", "200"))
    batch_size = int(os.environ.get("XLSTM_TS_BATCH_SIZE", "16"))  # Reduced batch size to save memory

    best_val_loss = float('inf')
    patience = int(os.environ.get("XLSTM_TS_PATIENCE", "40"))
    trigger_times = 0

    # DataLoader for batching
    train_loader = create_dataloader(train_x, train_y, batch_size, shuffle=True)
    val_loader = create_dataloader(val_x, val_y, batch_size, shuffle=False)

    # Define the loss function and optimiser
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimiser = optim.Adam(list(xlstm_stack.parameters()) + list(input_projection.parameters()) + list(output_projection.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10)  # Learning rate scheduler

    initial_lr = optimiser.param_groups[0]['lr']
    lr_reduced = False # Flag to track if learning rate has been reduced

    # Training loop
    for epoch in range(num_epochs):
        xlstm_stack.train()  # Set the model to training mode

        for batch_x, batch_y in train_loader:
            # Forward pass
            projected_input_data = input_projection(batch_x)
            xlstm_output = xlstm_stack(projected_input_data)
            predictions = output_projection(xlstm_output[:, -1, :])  # Use the last time step's output

            # Ensure the shapes match
            predictions = predictions.squeeze()
            batch_y = batch_y.squeeze()

            # Compute the loss
            loss = criterion(predictions, batch_y)

            # Backward pass and optimization
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(xlstm_stack.parameters(), max_norm=1.0)  # Apply gradient clipping to prevent the exploding gradient problem
            optimiser.step()

        # Validation step
        xlstm_stack.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                projected_input_data = input_projection(batch_x)
                xlstm_output = xlstm_stack(projected_input_data)
                predictions = output_projection(xlstm_output[:, -1, :])  # Use the last time step's output

                predictions = predictions.squeeze()
                batch_y = batch_y.squeeze()

                loss = criterion(predictions, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        # Print learning rate reduction only once
        if not lr_reduced and optimiser.param_groups[0]['lr'] < initial_lr:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Reducing learning rate to {optimiser.param_groups[0]["lr"]}')
            lr_reduced = True

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}, Validation Loss: {val_loss:.8f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(xlstm_stack.state_dict(), 'xlstm_ts_model.pth')
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping!')
                break

    print("Training complete!")

    return xlstm_stack, input_projection, output_projection

# -------------------------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------------------------

def evaluate_model(xlstm_stack, input_projection, output_projection, test_x):
    # Load the best model
    xlstm_stack.load_state_dict(torch.load('xlstm_ts_model.pth'))

    # Evaluate the model on the test set
    xlstm_stack.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        projected_input_data = input_projection(test_x)
        xlstm_output = xlstm_stack(projected_input_data)
        test_predictions = output_projection(xlstm_output[:, -1, :])  # Use the last time step's output

    return test_predictions
