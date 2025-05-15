# utils/torch_utils.py

import torch
import random
import numpy as np
import os
from pathlib import Path

def get_device():
    """
    Get the device to run the model (GPU or CPU).
    
    Returns:
        str: 'cuda' if GPU is available, otherwise 'cpu'
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name} ({gpu_count} device(s) available)")
    else:
        print("GPU not available, using CPU")
    return device

def load_model(model_path, model=None, device='cuda'):
    """
    Load a pre-trained model from a file path.
    
    Args:
        model_path (str): Path to the model's weights file
        model (torch.nn.Module, optional): Model architecture to load weights into
        device (str): Device to load the model on
        
    Returns:
        torch.nn.Module: Loaded model
    """
    print(f"Loading model from {model_path}...")
    
    # Create directory if it doesn't exist
    Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    
    try:
        # Case 1: Loading directly with torch.load
        if model is None:
            # Try to load entire model
            model = torch.load(model_path, map_location=device)
            
            # Check if model is a state dict, if so we need structure
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                else:
                    print("Error: model is a state dict but no model structure provided")
                    raise ValueError("Cannot load state dict without model structure")
        
        # Case 2: Loading state dict into provided model
        else:
            # Try to load state dict into model
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different formats of saved models
            if isinstance(checkpoint, dict):
                # If checkpoint has 'state_dict' key (PyTorch standard)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                # If checkpoint has 'model' key (YOLOv5 style)
                elif 'model' in checkpoint:
                    if isinstance(checkpoint['model'], dict):
                        state_dict = checkpoint['model']
                    else:
                        # Return the model directly if it's already a model
                        return checkpoint['model'].to(device)
                else:
                    # Assume the dict is the state dict itself
                    state_dict = checkpoint
            else:
                # Assume checkpoint is the state dict
                state_dict = checkpoint
            
            # Try to load state dict
            try:
                model.load_state_dict(state_dict)
            except:
                print("Warning: Strict loading failed, trying non-strict")
                model.load_state_dict(state_dict, strict=False)
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try YOLOv5 hub as fallback
        try:
            print("Attempting to load using torch hub...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            model = model.to(device)
            model.eval()
            print(f"Model loaded via torch hub on {device}")
            return model
        except Exception as hub_e:
            print(f"Torch hub loading failed: {hub_e}")
            raise RuntimeError(f"Failed to load model: {e}")

def save_model(model, model_path='weights/best.pt', optimizer=None, epoch=None, loss=None):
    """
    Save the model's weights to a file.
    
    Args:
        model (torch.nn.Module): Model to save
        model_path (str): Path to save the model
        optimizer (torch.optim.Optimizer, optional): Optimizer state
        epoch (int, optional): Current epoch number
        loss (float, optional): Current loss value
    """
    # Create directory if it doesn't exist
    Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint dict
    checkpoint = {
        'model': model.state_dict(),
    }
    
    # Add optional items if provided
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    
    # Save the checkpoint
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")

def set_seed(seed=42):
    """
    Set the seed for random number generators for reproducibility.
    
    Args:
        seed (int): Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def select_device(device=''):
    """
    Select the device to run the model based on input and availability.
    
    Args:
        device (str): Requested device, e.g., '0', '0,1,2,3', 'cpu'
        
    Returns:
        torch.device: Selected device
    """
    # Device preference: cuda > mps > cpu
    if device.lower() == 'cpu':
        return torch.device('cpu')
    
    # Check CUDA devices
    cuda = device.lower() != 'cpu' and torch.cuda.is_available()
    if cuda:
        # Default to all available CUDA devices if none specified
        if device == '':
            device = 'cuda:0'
        # Handle multi-GPU case
        elif ',' in device:
            device_ids = [int(x) for x in device.split(',')]
            device = f'cuda:{device_ids[0]}'
            os.environ['CUDA_VISIBLE_DEVICES'] = device
        else:
            device = f'cuda:{device}'
        
        # Ensure selected GPU exists
        device_id = int(device.split(':')[-1])
        assert device_id < torch.cuda.device_count(), f"GPU {device_id} not available, only {torch.cuda.device_count()} GPUs found"
        
        return torch.device(device)
    
    # Check for Apple MPS (Metal Performance Shaders)
    try:
        if hasattr(torch, 'has_mps') and torch.has_mps:
            return torch.device('mps')
    except:
        pass
    
    # Default to CPU
    print("CUDA not available, using CPU")
    return torch.device('cpu')

def enable_amp(model):
    """
    Enable automatic mixed precision for faster training with less memory usage.
    
    Args:
        model (torch.nn.Module): Model to convert
        
    Returns:
        tuple: (model, GradScaler)
    """
    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("Automatic Mixed Precision (AMP) enabled")
        return model, scaler
    except ImportError:
        print("AMP not available, continuing with full precision")
        return model, None
    
def model_info(model, verbose=False, img_size=640):
    """
    Print model information including parameter count, layer details, etc.
    
    Args:
        model (torch.nn.Module): Model to analyze
        verbose (bool): Whether to print detailed layer information
        img_size (int): Input size for calculating FLOPS
    """
    # Count number of parameters
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    
    print(f"Model Summary: {len(list(model.modules()))} layers, " +
          f"{n_p:,} parameters, {n_g:,} gradients")
    
    # Print detailed info if verbose
    if verbose:
         from prettytable import PrettyTable
        
        # Create table for model layers
         table = PrettyTable(["Layer", "Type", "Parameters"])
         table.align["Layer"] = "l"
         table.align["Type"] = "l"
         table.align["Parameters"] = "r"
        
         for name, module in model.named_modules():
            if not name:  # Skip the model itself
                continue
                
            param = sum(p.numel() for p in module.parameters())
            if param > 0:  # Only show layers with parameters
                table.add_row([name, module.__class__.__name__, f"{param:,}"])
        
                print(table)