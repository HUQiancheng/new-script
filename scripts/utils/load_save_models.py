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

def inference(model, dataloader, device, num_samples=4):
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
            print(f"Label shape: {labels.shape}, unique values: {torch.unique(labels)}")
            
            # Identify invalid labels and set them to IGNORE_INDEX
            invalid_mask = (labels >= 100) | (labels < 0)
            labels[invalid_mask] = -1
            
            outputs = model(inputs)
            print(f"Output shape: {outputs.shape}, unique values: {torch.unique(outputs)}")
            # Ensure the shapes match before calculating the loss
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = nn.functional.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            
            all_inputs.append(inputs.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(outputs.cpu())
    
    return all_inputs, all_labels, all_predictions
