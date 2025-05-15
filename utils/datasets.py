# utils/datasets.py

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_image(image_path, img_size=(640, 640)):
    """ 
    Load and resize an image for input to the model.
    
    Args:
        image_path (str): Path to the image file
        img_size (tuple): Target size to resize the image
        
    Returns:
        tuple: (original image, processed tensor)
    """
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Store original for visualization
    orig_img = img.copy() 
    
    # Process image for model input
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))    # Change HWC to CHW format
    img = np.expand_dims(img, axis=0)     # Add batch dimension
    img = torch.tensor(img).float()
    
    return orig_img, img

def load_annotations(annotations_path):
    """ 
    Load bounding box annotations for the images.
    
    Args:
        annotations_path (str): Path to annotation file
        
    Returns:
        list: List of annotations
    """
    annotations = []
    try:
        with open(annotations_path, 'r') as file:
            for line in file:
                # Parse line into a list of values
                values = line.strip().split()
                
                # YOLO format: [class_id, x_center, y_center, width, height]
                if len(values) == 5:  
                    class_id = int(values[0])
                    x_center, y_center = float(values[1]), float(values[2])
                    width, height = float(values[3]), float(values[4])
                    annotations.append([class_id, x_center, y_center, width, height])
                    
                # COCO-like format: [x1, y1, x2, y2, class_id]    
                elif len(values) >= 5:  
                    x1, y1, x2, y2 = map(float, values[:4])
                    class_id = int(values[4])
                    annotations.append([x1, y1, x2, y2, class_id])
    except Exception as e:
        print(f"Error loading annotations from {annotations_path}: {e}")
        return []
        
    return annotations

def split_data(image_dir, label_dir=None, test_size=0.2, val_size=0.1, random_state=42):
    """ 
    Split data into train, validation, and test sets.
    
    Args:
        image_dir (str): Directory containing image files
        label_dir (str, optional): Directory containing label files
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_images, val_images, test_images, train_labels, val_labels, test_labels)
    """
    # Get all image files
    image_files = [str(p) for p in Path(image_dir).glob('**/*') 
                  if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')]
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    # Create label files list if label_dir is provided
    label_files = None
    if label_dir:
        # Match each image file with its corresponding label file
        label_files = []
        for img_path in image_files:
            img_name = Path(img_path).stem
            label_path = str(Path(label_dir) / f"{img_name}.txt")
            label_files.append(label_path)
    
    # First split: separate test set
    if label_files:
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            image_files, label_files, test_size=test_size, random_state=random_state
        )
    else:
        train_val_images, test_images = train_test_split(
            image_files, test_size=test_size, random_state=random_state
        )
        train_val_labels, test_labels = None, None
    
    # Second split: separate validation set from training set
    # Adjust val_size to account for the proportion of the original dataset
    adjusted_val_size = val_size / (1 - test_size)
    
    if label_files:
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images, train_val_labels, test_size=adjusted_val_size, random_state=random_state
        )
    else:
        train_images, val_images = train_test_split(
            train_val_images, test_size=adjusted_val_size, random_state=random_state
        )
        train_labels, val_labels = None, None
    
    # Print split statistics
    print(f"Dataset split: {len(train_images)} training, {len(val_images)} validation, {len(test_images)} test images")
    
    if label_files:
        return train_images, val_images, test_images, train_labels, val_labels, test_labels
    else:
        return train_images, val_images, test_images

def image_augmentations(image, annotations=None, flip=True, rotate=True, brightness=True, contrast=True):
    """ 
    Apply basic augmentations to images and adjust annotations accordingly.
    
    Args:
        image (numpy.ndarray): Input image
        annotations (list, optional): List of bounding box annotations
        flip (bool): Apply horizontal flip
        rotate (bool): Apply rotation
        brightness (bool): Apply brightness adjustment
        contrast (bool): Apply contrast adjustment
        
    Returns:
        tuple: (augmented image, adjusted annotations)
    """
    h, w = image.shape[:2]
    adjusted_annotations = annotations.copy() if annotations else None
    
    # Horizontal flip
    if flip and np.random.rand() > 0.5:
        image = cv2.flip(image, 1)  # 1 for horizontal flip
        
        # Update annotations if provided
        if adjusted_annotations:
            for i, ann in enumerate(adjusted_annotations):
                if len(ann) == 5:  # [class_id, x_center, y_center, width, height]
                    # Flip x_center (keep width and height the same)
                    adjusted_annotations[i][1] = 1.0 - ann[1]
                elif len(ann) >= 5:  # [x1, y1, x2, y2, class_id]
                    # Flip x coordinates
                    x1, x2 = ann[0], ann[2]
                    adjusted_annotations[i][0] = w - x2
                    adjusted_annotations[i][2] = w - x1
    
    # Rotate image
    if rotate and np.random.rand() > 0.5:
        angle = np.random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Note: Rotating bounding boxes is more complex and omitted here
        # For real implementation, you would need to rotate each point of the bounding box
    
    # Brightness adjustment
    if brightness and np.random.rand() > 0.5:
        value = np.random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Contrast adjustment
    if contrast and np.random.rand() > 0.5:
        value = np.random.uniform(0.7, 1.3)
        mean = np.mean(image, axis=(0, 1))
        image = image.astype(np.float32)
        image = (image - mean) * value + mean
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
    
    return image, adjusted_annotations

def create_data_loader(images, labels=None, batch_size=16, img_size=(640, 640), augment=False):
    """
    Create a simple data loader that yields batches of processed images and labels.
    
    Args:
        images (list): List of image paths
        labels (list, optional): List of label paths
        batch_size (int): Batch size
        img_size (tuple): Image dimensions for model input
        augment (bool): Whether to apply augmentations
        
    Yields:
        tuple: (batch of images tensor, batch of labels tensor)
    """
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    
    n_batches = int(np.ceil(len(images) / batch_size))
    
    for i in range(n_batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            img_path = images[idx]
            
            # Load and process image
            orig_img, img_tensor = load_image(img_path, img_size)
            
            # Apply augmentations if enabled
            if augment:
                label_data = []
                if labels:
                    label_path = labels[idx]
                    label_data = load_annotations(label_path)
                
                aug_img, aug_labels = image_augmentations(orig_img, label_data)
                _, img_tensor = load_image(aug_img, img_size)
                batch_labels.append(aug_labels if aug_labels else [])
            elif labels:
                label_path = labels[idx]
                label_data = load_annotations(label_path)
                batch_labels.append(label_data)
            
            batch_images.append(img_tensor)
        
        # Stack images into batch tensor
        batch_images = torch.cat(batch_images, dim=0)
        
        if labels:
            yield batch_images, batch_labels
        else:
            yield batch_images