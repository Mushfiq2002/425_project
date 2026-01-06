"""
Utility functions for the music clustering pipeline.
"""
import os
import random
import logging
import numpy as np
import torch
from typing import Optional, Dict, Any
import json


def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logging.info(f"Random seed set to: {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the appropriate device (CPU or CUDA).
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    return device


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   path: str,
                   metadata: Optional[Dict[str, Any]] = None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        metadata: Additional metadata to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved to: {path}")


def load_checkpoint(model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer],
                   path: str,
                   device: torch.device) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optional optimizer
        path: Path to checkpoint
        device: Device to load model to
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logging.info(f"Checkpoint loaded from: {path}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: Dict, path: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def check_file_exists(path: str, name: str = "File") -> bool:
    """
    Check if file exists and log appropriate message.
    
    Args:
        path: Path to check
        name: Name of file for logging
        
    Returns:
        True if exists, False otherwise
    """
    exists = os.path.exists(path)
    if exists:
        logging.info(f"{name} found: {path}")
    else:
        logging.warning(f"{name} not found: {path}")
    return exists


def normalize_features(features: np.ndarray, 
                      method: str = 'standard',
                      axis: int = 0) -> np.ndarray:
    """
    Normalize features.
    
    Args:
        features: Feature array
        method: 'standard' (z-score) or 'minmax'
        axis: Axis along which to normalize
        
    Returns:
        Normalized features
    """
    if method == 'standard':
        mean = np.mean(features, axis=axis, keepdims=True)
        std = np.std(features, axis=axis, keepdims=True)
        return (features - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(features, axis=axis, keepdims=True)
        max_val = np.max(features, axis=axis, keepdims=True)
        return (features - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

