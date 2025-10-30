print("loading libraries - this may take a minute")
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import torch
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
import random
import tqdm
print("libraries loaded")

from Danial import BaseModel, LayerFreezing, model_utils, data_loc, torchlight, DETRloss, load_model, validate

checkpoint_dir = data_loc.models
model = BaseModel.load()
LayerFreezing.freeze_transformers(model)
LayerFreezing.freeze_backbone(model)

model = model_utils.model_(model, data_loc.num_classes())
model.to(torchlight.transfer())
criterion = DETRloss.loss_fn(model)





import os
import logging
from datetime import datetime

# Set up logging configuration
log_file = os.path.join(checkpoint_dir, 'log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will still print to console
    ]
)
model, optimizer, start_epoch, num_epochs, checkpoint_dir = load_model.load(model, torchlight.transfer(), checkpoint_dir)
# Log training start information
logging.info("="*50)
logging.info("TRAINING SESSION STARTED")
logging.info(f"Start epoch: {start_epoch}")
logging.info(f"Total epochs: {num_epochs}")
logging.info(f"device(): {torchlight.transfer()}")
logging.info("="*50)

for epoch in range(start_epoch, num_epochs): # start from the latest epoch and end with the end epoch(300)
    try: # -------------------------------------------------------------------------------------------------------------------------------------------------------------------- this is to handle errors
        logging.info(f"Starting Epoch {epoch+1}/{num_epochs}")
        
        model.train() # ------------------------------------------------------------------------------------------------------------------------------------------------------ puts the model into training mode
        running_loss = 0.0 # -------------------------------------------------------------------------------------------------------------------------------------------------- sets "cumulative loss" to 0
        batch_count = 0
        
        # Wrap the DataLoader with tqdm for a progress bar
        for images, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move images and targets to the torchlight.transfer() (GPU if available)
            images = [img.to(torchlight.transfer()) for img in images]
            targets = [{k: v.to(torchlight.transfer()) for k, v in t.items()} for t in targets]
            outputs = model(images)


            
            losses = criterion(outputs, targets) # Calculate total loss using detrloss
            loss = losses['loss_total']
           # print("---------------------------------------------------------------------------------------------------------------------------")
           #print(f"the loss is {loss}")
           # print("---------------------------------------------------------------------------------------------------------------------------")
           # print(f"the losses is {losses}") 
           # print("---------------------------------------------------------------------------------------------------------------------------")
           # print(f"the loss total is {losses['loss_total']}") 
           # print(f"with type {type(losses['loss_total'])}")
           # print(losses['loss_total'].requires_grad)

           # print("---------------------------------------------------------------------------------------------------------------------------")
           # print("---------------------------------------------------------------------------------------------------------------------------")
           # print("---------------------------------------------------------------------------------------------------------------------------")
            
            
            optimizer.zero_grad()
           # print("optimising done")
            loss.backward()
           # print("backwards done")
            optimizer.step()
           # print("stepping done")
            running_loss += loss.item()
            batch_count += 1
        
        # Calculate average training loss for this epoch
        avg_train_loss = running_loss / len(data_loc.train)
        
        # Log training results
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
        
        # Save the model checkpoint after each successful epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved to {checkpoint_path}")
        print(f"Model saved to {checkpoint_path}")
        
        # Run validation and log results
        val_loss = validate.validate_model(model, data_loc.val, criterion, torchlight.transfer(), checkpoint_dir)
        if val_loss is not None:
            logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
        else:
            logging.warning(f"Epoch [{epoch+1}/{num_epochs}] - Validation returned None")
        
        # Log epoch completion
        logging.info(f"Epoch {epoch+1} completed successfully")
        logging.info("-" * 30)
        
    except Exception as e:
        error_msg = f"An error occurred during epoch {epoch+1}: {e}"
        logging.error(error_msg)
        print(error_msg)
        
        logging.info("Attempting to load the last saved checkpoint and stopping training.")
        print("Attempting to load the last saved checkpoint and stopping training.")
        
        # Find the last saved checkpoint before the error occurred
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            last_checkpoint = max([os.path.join(checkpoint_dir, f) for f in checkpoints], key=os.path.getctime)
            logging.info(f"Loading checkpoint from: {last_checkpoint}")
            print(f"Loading checkpoint from: {last_checkpoint}")
            try:
                model.load_state_dict(torch.load(last_checkpoint))
                logging.info("Successfully loaded last checkpoint.")
                print("Successfully loaded last checkpoint.")
            except Exception as load_error:
                error_msg = f"Error loading checkpoint: {load_error}"
                logging.error(error_msg)
                print(error_msg)
        else:
            logging.warning("No checkpoints found to load.")
            print("No checkpoints found to load.")
        
        logging.info("Training stopped due to error.")
        break # Stop the training loop after encountering an error

# Log training completion
logging.info("="*50)
logging.info("TRAINING SESSION ENDED")
logging.info(f"Final epoch reached: {epoch+1}")
logging.info("="*50)

# After training (or interruption), move the model back to CPU if needed for inference or saving
# model.to('cpu')
# check epoch 27 and 74