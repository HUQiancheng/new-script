import os
import torch
import torch.nn as nn

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir, filename='checkpoint.pth.tar'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth.tar')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

def inference(model, dataloader, device, checkpoint_path, num_samples=4):
    checkpoints = torch.load(checkpoint_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()
    all_inputs = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for i, images in enumerate(dataloader):
            if i >= num_samples:
                break
            inputs = images['image'].to(device)
            labels = images['label'].to(device)
            
            
            # Split the inputs into two for each image along the second dimension
            inputs1, inputs2 = torch.split(inputs, 1, dim=1)  # Shape: (batch_size, 1, 3, height, width)
        
            # Remove the second dimension
            inputs1 = inputs1.squeeze(1)  # Shape: (batch_size, 3, height, width)
            inputs2 = inputs2.squeeze(1)  # Shape: (batch_size, 3, height, width)
        
            # Split the labels into two for each image along the second dimension
            labels1, labels2 = torch.split(labels, 1, dim=1)  # Shape: (batch_size, 1, height, width)
            
            # Remove the second dimension
            labels1 = labels1.squeeze(1)  # Shape: (batch_size, height, width)
            labels2 = labels2.squeeze(1)  # Shape: (batch_size, height, width)
            

            # Identify invalid labels and set them to IGNORE_INDEX
            invalid_mask1 = (labels1 >= 100) | (labels1 < 0)
            invalid_mask2 = (labels2 >= 100) | (labels2 < 0)
            labels1[invalid_mask1] = -1
            labels2[invalid_mask2] = -1

            outputs1 = model(inputs1)
            outputs2 = model(inputs2)
            
            # Ensure the shapes match before calculating the loss
            if outputs1.shape[-2:] != labels1.shape[-2:]:
                outputs1 = nn.functional.interpolate(outputs1, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            if outputs2.shape[-2:] != labels1.shape[-2:]:
                outputs2 = nn.functional.interpolate(outputs2, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            inputs_stack = torch.stack((inputs1,inputs2)).permute((1, 0, 2, 3, 4))
            labels_stack = torch.stack((labels1,labels2)).permute((1, 0, 2, 3))
            predictions_stack = torch.stack((outputs1,outputs2)).permute((1, 0, 2, 3, 4))
            print(f"Label shape: {labels_stack.shape}, unique values: {torch.unique(labels_stack)}")
            print(f"Prediction shape: {predictions_stack.shape}, unique values: {torch.unique(predictions_stack)}")
            all_inputs.append(inputs_stack.cpu())
            all_labels.append(labels_stack.cpu())
            all_predictions.append(predictions_stack.cpu())
    
    return all_inputs, all_labels, all_predictions

