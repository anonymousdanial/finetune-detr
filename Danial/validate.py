from tqdm import tqdm
import torch
import os
torch.cuda.empty_cache()

#validation call

def validate_model(model, dataloader, loss_fn, device, checkpoint_dir):
    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoints found in the specified directory.")
        return None # Or raise an error

    latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in checkpoints], key=os.path.getctime)
    print(f"Loading model from: {latest_checkpoint}")
    model.load_state_dict(torch.load(latest_checkpoint))

    model.eval() # Set the model to evaluation mode
    running_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            # Use the correct loss function (DETRLoss)
            loss = loss_fn(outputs, targets)['loss_total']


            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss
