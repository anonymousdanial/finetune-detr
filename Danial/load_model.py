from tqdm import tqdm
import torch
import os
import torch.optim as optim # Import the optim module

def load(model2, device, path):
    num_epochs = 300
    # Move the model to the device (GPU if available)
    model2.to(device)

    # Define a directory to save checkpoints
    checkpoint_dir = path
    os.makedirs(checkpoint_dir, exist_ok=True) # just make the dir if not there

    start_epoch = 0
    optimizer = None  # Initialize optimizer to None

    # Check for existing checkpoints to resume training
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoints:
        print("Found existing checkpoints. Resuming training.")
        # Find the latest checkpoint
        latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in checkpoints], key=os.path.getctime) # ---- if folder has .pth file, use the one trained latest
        print(f"Loading checkpoint from: {latest_checkpoint}")
        model2.load_state_dict(torch.load(latest_checkpoint, map_location=torch.device('cpu'))) # ------------------ load the thing
        start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) # ---------------------------------------- set the epoch to the last trained epoch
        print(f"Resuming from epoch {start_epoch + 1}")

        optimizer = optim.AdamW(model2.parameters(), lr=1e-5)
        return model2, optimizer, start_epoch, num_epochs, checkpoint_dir

    else:
        print("No existing checkpoints found. Starting training from scratch.")
        optimizer = optim.AdamW(model2.parameters(), lr=1e-5)
